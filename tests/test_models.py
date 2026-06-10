"""Sklearn adapter behavior: NaN handling, calibration bounds, persistence."""

import numpy as np
import pytest

from predictor import FeatureSchema, ValidationEngine
from predictor.models import (
    GradientBoostingPredictor,
    RandomForestPredictor,
    load_predictor,
)

SCHEMA = FeatureSchema([(f"test_{i}", "value") for i in range(6)])


def synthetic_data(n=400, seed=0):
    """Failure is driven by feature 0; ~30% of cells are NaN placeholders."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(SCHEMA)))
    y = (X[:, 0] > 0.3).astype(int)
    nan_mask = rng.random(X.shape) < 0.3
    nan_mask[:, 0] = rng.random(n) < 0.1  # keep the signal mostly observed
    X[nan_mask] = np.nan
    return X, y


@pytest.fixture(scope="module", params=[RandomForestPredictor, GradientBoostingPredictor])
def trained(request):
    X, y = synthetic_data()
    cls = request.param
    params = {"n_estimators": 30} if cls is RandomForestPredictor else {"max_iter": 50}
    return cls.train(
        X, y, schema_hash=SCHEMA.schema_hash, calibration="sigmoid", **params
    )


def test_probability_bounds_and_signal(trained):
    failing = [3.0] + [np.nan] * 5   # strong feature-0 signal, rest unseen
    passing = [-3.0] + [np.nan] * 5
    p_fail = trained.predict_failure_probability(failing)
    p_pass = trained.predict_failure_probability(passing)
    assert 0.0 <= p_pass < p_fail <= 1.0
    assert p_fail > 0.6 and p_pass < 0.4


def test_all_nan_state_is_handled(trained):
    p = trained.predict_failure_probability([np.nan] * len(SCHEMA))
    assert 0.0 <= p <= 1.0


def test_save_load_roundtrip(trained, tmp_path):
    path = str(tmp_path / "model.joblib")
    trained.save(path)
    loaded = load_predictor(path)
    assert type(loaded) is type(trained)
    assert loaded.schema_hash == SCHEMA.schema_hash
    state = [0.5] + [np.nan] * 5
    assert loaded.predict_failure_probability(state) == pytest.approx(
        trained.predict_failure_probability(state)
    )


def test_loaded_model_works_in_engine(trained, tmp_path):
    path = str(tmp_path / "model.joblib")
    trained.save(path)
    engine = ValidationEngine(SCHEMA, load_predictor(path), threshold=0.99)
    assert 0.0 <= engine.ingest("test_1", "value", 0.2) <= 1.0


def test_engine_rejects_stale_schema(trained, tmp_path):
    path = str(tmp_path / "model.joblib")
    trained.save(path)
    other_schema = FeatureSchema([("test_x", "value")])
    with pytest.raises(ValueError, match="different feature schema"):
        ValidationEngine(other_schema, load_predictor(path))


def test_single_class_labels_rejected():
    X = np.zeros((20, len(SCHEMA)))
    with pytest.raises(ValueError, match="both classes"):
        RandomForestPredictor.train(X, np.zeros(20, dtype=int))


def test_wrong_kind_load_rejected(trained, tmp_path):
    path = str(tmp_path / "model.joblib")
    trained.save(path)
    other = (
        GradientBoostingPredictor
        if isinstance(trained, RandomForestPredictor)
        else RandomForestPredictor
    )
    with pytest.raises(ValueError, match="contains a"):
        other.load(path)
