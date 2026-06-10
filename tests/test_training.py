"""End-to-end MLOps pipeline: JSONL records → dataset → train → promote."""

import json

import numpy as np
import pytest

from predictor import FeatureSchema, JsonlRecorder, StateVector
from predictor.models import RandomForestPredictor
from predictor.training import (
    ModelRegistry,
    build_dataset,
    drift_report,
    should_promote,
    temporal_split,
    train_candidate,
)

SCHEMA = FeatureSchema(
    [
        ("test_voltage", "vout"),
        ("test_ripple", "mv"),
        ("test_thermal", "temp_c"),
        ("test_current", "amps"),
    ]
)


def write_runs(path, n_runs=40, seed=0):
    """Simulate suite runs: high temp_c drives failure, rest is noise."""
    rng = np.random.default_rng(seed)
    recorder = JsonlRecorder(path)
    for i in range(n_runs):
        failed = i % 2 == 1
        readings = [
            ("test_voltage", "vout", rng.normal(3.3, 0.05)),
            ("test_ripple", "mv", rng.normal(20, 5)),
            ("test_thermal", "temp_c", rng.normal(95 if failed else 45, 3)),
            ("test_current", "amps", rng.normal(1.2, 0.1)),
        ]
        state = StateVector(SCHEMA)
        for test_id, metric, value in readings:
            state.update(test_id, metric, value)
            recorder.record(
                test_id, metric, state.snapshot(), None, SCHEMA.schema_hash
            )
        recorder.finalize(suite_passed=not failed)


@pytest.fixture(scope="module")
def records_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("records") / "runs.jsonl"
    write_runs(path)
    return path


@pytest.fixture(scope="module")
def dataset(records_file):
    return build_dataset([records_file], SCHEMA.schema_hash)


def test_dataset_shape_and_labels(dataset):
    assert dataset.X.shape == (40 * 4, len(SCHEMA))
    assert dataset.n_runs == 40
    assert set(np.unique(dataset.y)) == {0, 1}
    # first snapshot of every run has exactly one observed feature
    first_rows = dataset.X[::4]
    assert (np.isnan(first_rows).sum(axis=1) == len(SCHEMA) - 1).all()


def test_dataset_rejects_unknown_schema(records_file):
    with pytest.raises(ValueError, match="no finalized runs"):
        build_dataset([records_file], "f" * 64)


def test_unfinalized_run_dropped_with_warning(tmp_path):
    path = tmp_path / "partial.jsonl"
    write_runs(path, n_runs=2)
    with path.open("a", encoding="utf-8") as fh:  # crashed run, never labeled
        fh.write(
            json.dumps(
                {
                    "event": "ingest",
                    "test_id": "test_voltage",
                    "metric": "vout",
                    "state": [3.3, None, None, None],
                    "probability": None,
                    "schema_hash": SCHEMA.schema_hash,
                    "ts": 1.0,
                }
            )
            + "\n"
        )
    with pytest.warns(UserWarning, match="unfinalized"):
        ds = build_dataset([path], SCHEMA.schema_hash)
    assert ds.n_runs == 2


def test_temporal_split_holds_out_newest_runs(dataset):
    train_mask, holdout_mask = temporal_split(dataset, holdout_fraction=0.25)
    assert not (train_mask & holdout_mask).any()
    assert dataset.run_id[train_mask].max() < dataset.run_id[holdout_mask].min()


def test_train_candidate_learns_and_reports(dataset):
    result = train_candidate(
        dataset,
        RandomForestPredictor,
        threshold=0.7,
        holdout_fraction=0.25,
        calibration="sigmoid",
        n_estimators=30,
    )
    m = result.metrics
    assert result.n_train_runs == 30 and result.n_holdout_runs == 10
    # temp_c separates classes cleanly: most failing holdout runs are caught
    # (calibration squashes probabilities, so 100% is not guaranteed), no
    # passing run is wrongly aborted, and time is actually saved
    assert m["early_catch_rate"] >= 0.8
    assert m["false_abort_rate"] == 0.0
    assert m["mean_fraction_saved"] > 0.0
    # half the snapshots predate the discriminating temp_c reading and are
    # inherently 50/50; just require beating the uninformed 0.25 baseline
    assert m["brier"] < 0.25


def test_registry_versioning_and_promotion(dataset, tmp_path):
    result = train_candidate(
        dataset,
        RandomForestPredictor,
        threshold=0.7,
        calibration="sigmoid",
        n_estimators=20,
    )
    registry = ModelRegistry(tmp_path / "registry")
    assert registry.latest(SCHEMA.schema_hash) is None
    assert should_promote(result.metrics, registry.latest_metrics(SCHEMA.schema_hash))

    first = registry.save(result.predictor, result.metrics)
    assert first.name == f"model_v1_{SCHEMA.schema_hash[:8]}.joblib"
    assert registry.latest_metrics(SCHEMA.schema_hash) == result.metrics

    second = registry.save(result.predictor, result.metrics)
    assert second.name.startswith("model_v2_")
    assert registry.latest(SCHEMA.schema_hash) == second


def test_should_promote_caps_false_aborts():
    good = {"false_abort_rate": 0.0, "mean_fraction_saved": 0.5, "brier": 0.1}
    reckless = {"false_abort_rate": 0.2, "mean_fraction_saved": 0.9, "brier": 0.05}
    assert should_promote(good, None)
    assert not should_promote(reckless, None)
    assert not should_promote(
        {"false_abort_rate": 0.0, "mean_fraction_saved": 0.3, "brier": 0.1}, good
    )


def test_drift_report_flags_shift():
    rng = np.random.default_rng(1)
    baseline = rng.normal(0, 1, size=(500, 4))
    stable = drift_report(baseline, rng.normal(0, 1, size=(500, 4)))
    assert not stable["retrain_recommended"]

    shifted = baseline.copy()
    shifted[:, 2] += 3.0  # thermal characteristics changed
    moved = drift_report(baseline, shifted)
    assert moved["retrain_recommended"]
    assert 2 in moved["flagged_features"]


def test_drift_report_flags_missing_rate_shift():
    rng = np.random.default_rng(2)
    baseline = rng.normal(0, 1, size=(500, 4))
    recent = rng.normal(0, 1, size=(500, 4))
    recent[rng.random(500) < 0.5, 3] = np.nan  # a test stopped running
    report = drift_report(baseline, recent)
    assert report["retrain_recommended"]
    assert 3 in report["flagged_features"]
