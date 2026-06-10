"""Core engine behavior with a scripted stub predictor (no ML dependency)."""

import math

import pytest

from predictor import (
    BoundsRule,
    FailurePredictor,
    FeatureSchema,
    MLThresholdAbort,
    RuleAbort,
    ValidationEngine,
)

SCHEMA = FeatureSchema(
    [
        ("test_voltage_rail", "vout"),
        ("test_voltage_rail", "ripple_mv"),
        ("test_thermal", "temp_c"),
        ("test_current_draw", "amps"),
    ]
)


class ScriptedPredictor(FailurePredictor):
    """Returns pre-scripted probabilities in order, ignoring the state."""

    def __init__(self, probabilities):
        self._probs = iter(probabilities)
        self.seen_states = []

    def predict_failure_probability(self, state):
        self.seen_states.append(list(state))
        return next(self._probs)


def test_returns_probability_and_updates_state():
    engine = ValidationEngine(SCHEMA, ScriptedPredictor([0.2]), threshold=0.9)
    p = engine.ingest("test_voltage_rail", "vout", 3.31)
    assert p == 0.2
    snap = engine.state.snapshot()
    assert snap[0] == 3.31
    assert all(math.isnan(v) for v in snap[1:])


def test_ml_threshold_abort_carries_context():
    engine = ValidationEngine(SCHEMA, ScriptedPredictor([0.95]), threshold=0.85)
    with pytest.raises(MLThresholdAbort) as exc:
        engine.ingest("test_thermal", "temp_c", 104.0)
    assert exc.value.probability == 0.95
    assert exc.value.threshold == 0.85


def test_rule_fires_before_ml_and_before_state_update():
    predictor = ScriptedPredictor([0.0])
    engine = ValidationEngine(
        SCHEMA,
        predictor,
        rules=[BoundsRule("vout", max_value=5.5)],
    )
    with pytest.raises(RuleAbort):
        engine.ingest("test_voltage_rail", "vout", 12.0)
    assert predictor.seen_states == []  # ML was bypassed
    assert math.isnan(engine.state.snapshot()[0])  # spike not recorded


def test_warmup_gate_skips_prediction_until_coverage():
    predictor = ScriptedPredictor([0.99])
    engine = ValidationEngine(SCHEMA, predictor, min_coverage=0.5)
    assert engine.ingest("test_voltage_rail", "vout", 3.3) is None
    # second of four features -> coverage 0.5 -> prediction runs
    with pytest.raises(MLThresholdAbort):
        engine.ingest("test_thermal", "temp_c", 45.0)


def test_unknown_feature_rejected():
    engine = ValidationEngine(SCHEMA, ScriptedPredictor([0.1]))
    with pytest.raises(KeyError):
        engine.ingest("test_unknown", "vout", 1.0)


def test_schema_hash_mismatch_rejected():
    class StaleModel(ScriptedPredictor):
        @property
        def schema_hash(self):
            return "0" * 64

    with pytest.raises(ValueError, match="different feature schema"):
        ValidationEngine(SCHEMA, StaleModel([]))
