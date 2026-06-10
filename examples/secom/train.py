"""Train fail-fast models on real SECOM data and promote the winner.

Pipeline: raw SECOM -> JSONL run records (via the real recorder) -> dataset
-> temporal split -> train RF + GB (class-balanced, calibrated) -> threshold
sweep on the holdout -> promote the winner into the model registry.

Run download.py first. Then:  python examples/secom/train.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    HISTORY,
    REGISTRY_DIR,
    build_schema,
    load_raw,
    select_sensors,
    unit_readings,
)

from predictor import JsonlRecorder, StateVector
from predictor.models import GradientBoostingPredictor, RandomForestPredictor
from predictor.training import (
    ModelRegistry,
    build_dataset,
    predict_rows,
    should_promote,
    simulate_fail_fast,
    temporal_split,
)
from predictor.training.evaluate import brier_score

LOG = logging.getLogger("secom.train")

HOLDOUT_FRACTION = 0.2
MAX_FALSE_ABORT = 0.05
# SECOM's signal is weak and the base failure rate is ~6.6%, so calibrated
# probabilities stay low; sweep well below 0.5. A calibrated P=0.20 is still
# ~3x the base rate — "user-defined threshold" is doing real work here.
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.025), 3)


def convert(schema, keep) -> None:
    """Write every unit through the real recorder, oldest lot first.

    One snapshot per station (not per sensor) keeps the JSONL tractable:
    1,567 units x ~25 stations instead of x ~450 sensors.
    """
    if HISTORY.exists():
        LOG.info("history already converted: %s", HISTORY)
        return
    X, failed, ts = load_raw()
    order = np.argsort(ts)  # recorder timestamps then preserve real lot order
    recorder = JsonlRecorder(HISTORY)
    for n, unit in enumerate(order, 1):
        state = StateVector(schema)
        current_station = None
        for test_id, metric, value in unit_readings(X[unit], keep):
            if current_station not in (None, test_id):
                recorder.record(current_station, "checkpoint", state.snapshot(),
                                None, schema.schema_hash)
            current_station = test_id
            state.update(test_id, metric, value)
        if current_station is not None:
            recorder.record(current_station, "checkpoint", state.snapshot(),
                            None, schema.schema_hash)
        recorder.finalize(suite_passed=not failed[unit])
        if n % 250 == 0:
            LOG.info("  converted %d/%d units", n, len(order))
    LOG.info("history written: %s", HISTORY)


def sweep(dataset, holdout_mask, probabilities) -> tuple[float, dict]:
    """Pick the threshold maximizing time saved under the false-abort cap."""
    full = np.zeros(len(dataset.y))
    full[holdout_mask] = probabilities
    best_t, best = None, None
    for t in THRESHOLDS:
        m = simulate_fail_fast(dataset, holdout_mask, full, float(t))
        if (m["false_abort_rate"] or 0.0) > MAX_FALSE_ABORT:
            continue
        if best is None or m["mean_fraction_saved"] > best["mean_fraction_saved"]:
            best_t, best = float(t), m
    best["brier"] = brier_score(probabilities, dataset.y[holdout_mask])
    return best_t, best


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("predictor").setLevel(logging.WARNING)

    X, failed, _ = load_raw()
    keep = select_sensors(X)
    schema = build_schema(keep)
    LOG.info("SECOM: %d units (%d failed), %d/%d sensors kept -> %d stations",
             len(X), int(failed.sum()), len(keep), X.shape[1],
             len({t for t, _ in schema.features}))

    convert(schema, keep)
    dataset = build_dataset([HISTORY], schema.schema_hash)
    train_mask, holdout_mask = temporal_split(dataset, HOLDOUT_FRACTION)
    LOG.info("dataset: %d rows / %d runs; holdout = newest %d runs (%d failing)",
             len(dataset.y), dataset.n_runs,
             len(np.unique(dataset.run_id[holdout_mask])),
             len(np.unique(dataset.run_id[holdout_mask & (dataset.y == 1)])))

    results = {}
    for cls, params in (
        (RandomForestPredictor, {"n_estimators": 200, "class_weight": "balanced"}),
        (GradientBoostingPredictor, {"class_weight": "balanced"}),
    ):
        predictor = cls.train(
            dataset.X[train_mask], dataset.y[train_mask],
            schema_hash=schema.schema_hash, calibration="sigmoid", **params,
        )
        probs = predict_rows(predictor, dataset.X[holdout_mask])
        y_hold = dataset.y[holdout_mask]
        LOG.info("  holdout P(fail): failing rows mean=%.3f max=%.3f | "
                 "passing rows mean=%.3f p99=%.3f",
                 probs[y_hold == 1].mean(), probs[y_hold == 1].max(),
                 probs[y_hold == 0].mean(), np.percentile(probs[y_hold == 0], 99))
        threshold, metrics = sweep(dataset, holdout_mask, probs)
        results[cls.kind] = (predictor, threshold, metrics)
        LOG.info("%-28s best t=%.2f  catch=%.0f%%  saved=%.0f%%  "
                 "false_aborts=%.1f%%  brier=%.3f",
                 cls.__name__, threshold,
                 100 * (metrics["early_catch_rate"] or 0),
                 100 * metrics["mean_fraction_saved"],
                 100 * (metrics["false_abort_rate"] or 0),
                 metrics["brier"])

    rf, gb = results["random_forest"], results["hist_gradient_boosting"]
    predictor, threshold, metrics = gb if should_promote(gb[2], rf[2]) else rf
    LOG.info("promoted: %s at threshold %.2f", type(predictor).__name__, threshold)

    registry = ModelRegistry(REGISTRY_DIR)
    artifact = registry.save(predictor, metrics, extra={"threshold": threshold})
    LOG.info("registered -> %s", artifact)


if __name__ == "__main__":
    main()
