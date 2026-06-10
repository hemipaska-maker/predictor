"""Pytest plugin behavior, exercised through pytester sub-runs."""

import json

pytest_plugins = ["pytester"]

FACTORY = """
import predictor

class TempPredictor(predictor.FailurePredictor):
    def predict_failure_probability(self, state):
        temp = state[1]
        if temp != temp:  # NaN -> temp not measured yet
            return 0.10
        return 0.95 if temp > 100 else 0.05

def make_engine():
    schema = predictor.FeatureSchema([
        ("test_volt", "vout"),
        ("test_temp", "temp_c"),
        ("test_curr", "amps"),
    ])
    return predictor.ValidationEngine(
        schema,
        TempPredictor(),
        threshold=0.85,
        recorder=predictor.JsonlRecorder("records.jsonl"),
    )
"""

SUITE = """
def test_volt(validation_engine):
    assert validation_engine.ingest("test_volt", "vout", 3.3) < 0.5

def test_temp(validation_engine):
    validation_engine.ingest("test_temp", "temp_c", 130.0)  # crosses threshold

def test_curr(validation_engine):
    validation_engine.ingest("test_curr", "amps", 1.0)
"""


def run_suite(pytester, temp_value=130.0):
    pytester.makepyfile(engine_factory=FACTORY, test_suite=SUITE.replace("130.0", str(temp_value)))
    pytester.syspathinsert()
    return pytester.runpytest(
        "-p", "predictor.integrations.pytest_plugin",
        "--hve-engine", "engine_factory:make_engine",
        "-v",
    )


def test_abort_fails_test_and_stops_session(pytester):
    result = run_suite(pytester)
    # test_temp fails with the abort; test_curr never starts
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*validation engine abort*85*"])
    assert "test_curr" not in result.stdout.str() or "test_curr" not in [
        line for line in result.stdout.lines if "PASSED" in line
    ]


def test_run_is_finalized_for_training_data(pytester):
    run_suite(pytester)
    events = [
        json.loads(line)
        for line in (pytester.path / "records.jsonl").read_text().splitlines()
    ]
    assert events[-1] == {**events[-1], "event": "finalize", "suite_passed": False}


def test_healthy_suite_runs_to_completion(pytester):
    result = run_suite(pytester, temp_value=42.0)
    result.assert_outcomes(passed=3)
    events = [
        json.loads(line)
        for line in (pytester.path / "records.jsonl").read_text().splitlines()
    ]
    assert events[-1]["suite_passed"] is True


def test_missing_configuration_skips(pytester):
    pytester.makepyfile(test_suite=SUITE)
    result = pytester.runpytest("-p", "predictor.integrations.pytest_plugin")
    result.assert_outcomes(skipped=3)


INGEST_SUITE = """
import pytest

def test_volt(ingest):
    # the helper returns the value unchanged for inline assertions
    assert ingest("vout", 3.3) == 3.3

@pytest.mark.parametrize("amps", [0.9, 1.1])
def test_curr(ingest, amps):
    # parametrization suffix is stripped: test_id stays 'test_curr'
    ingest("amps", amps)

def test_temp(ingest):
    ingest("temp_c", 130.0)  # crosses the threshold
"""


def test_ingest_fixture_maps_test_names_and_aborts(pytester):
    pytester.makepyfile(engine_factory=FACTORY, test_suite=INGEST_SUITE)
    pytester.syspathinsert()
    result = pytester.runpytest(
        "-p", "predictor.integrations.pytest_plugin",
        "--hve-engine", "engine_factory:make_engine",
    )
    # volt + both curr params pass; temp aborts the session
    result.assert_outcomes(passed=3, failed=1)
