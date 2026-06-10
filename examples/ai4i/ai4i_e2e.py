"""Real-data fail-fast demo on the UCI AI4I 2020 predictive-maintenance set.

10,000 machine cycles, 339 failures (CC BY 4.0). Each cycle becomes a 6-step
"suite run": setup type, ambient temp, process temp, spindle speed, torque,
tool wear — ingested in that order, so failure signatures that live in the
torque/wear readings can abort a run before it completes.

Stages (all in one script, artifacts in a local temp dir):
  1. download   - fetch ai4i2020.csv from the UCI archive
  2. convert    - replay every cycle through the real JSONL recorder
  3. train      - RF + GB candidates, temporal split, threshold sweep,
                  promote the winner under the 5% false-abort cap
  4. replay     - feed holdout units through a live ValidationEngine:
                  known failures must abort; replaying twice must give the
                  identical trajectory (determinism invariant)

Run:  python examples/ai4i/ai4i_e2e.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from predictor import (
    FeatureSchema,
    JsonlRecorder,
    MLThresholdAbort,
    StateVector,
    ValidationEngine,
)
from predictor.models import GradientBoostingPredictor, RandomForestPredictor, load_predictor
from predictor.training import (
    ModelRegistry,
    build_dataset,
    predict_rows,
    should_promote,
    simulate_fail_fast,
    temporal_split,
)
from predictor.training.evaluate import brier_score

LOG = logging.getLogger("ai4i")

DATA_DIR = Path(os.environ.get("AI4I_DATA_DIR", Path(tempfile.gettempdir()) / "ai4i_predictor"))
CSV_PATH = DATA_DIR / "ai4i2020.csv"
HISTORY = DATA_DIR / "history.jsonl"
REGISTRY_DIR = DATA_DIR / "registry"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"

# The "suite": one reading per pseudo-test, in physical measurement order.
# Tool wear comes last — overstrain/wear failures need it, which is exactly
# what makes mid-run prediction non-trivial.
SCHEMA = FeatureSchema([
    ("setup", "type"),              # product quality variant L/M/H -> 0/1/2
    ("ambient", "air_temp_k"),
    ("process", "temp_k"),
    ("spindle", "speed_rpm"),
    ("spindle", "torque_nm"),
    ("tool", "wear_min"),
])

HOLDOUT_FRACTION = 0.2
MAX_FALSE_ABORT = 0.05
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.025), 3)
N_PASSING_SAMPLE = 60


def download() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_PATH.exists():
        LOG.info("dataset present: %s", CSV_PATH)
        return
    LOG.info("downloading %s ...", URL)
    urllib.request.urlretrieve(URL, CSV_PATH)
    LOG.info("  -> %s (%d bytes)", CSV_PATH, CSV_PATH.stat().st_size)


def load_units() -> tuple[np.ndarray, np.ndarray]:
    """Return (X[n,6] readings in schema order, failed[n]); rows in UDI order."""
    type_code = {"L": 0.0, "M": 1.0, "H": 2.0}
    X, failed = [], []
    with CSV_PATH.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            X.append([
                type_code[row["Type"]],
                float(row["Air temperature [K]"]),
                float(row["Process temperature [K]"]),
                float(row["Rotational speed [rpm]"]),
                float(row["Torque [Nm]"]),
                float(row["Tool wear [min]"]),
            ])
            failed.append(row["Machine failure"] == "1")
    return np.asarray(X), np.asarray(failed)


def convert(X: np.ndarray, failed: np.ndarray) -> None:
    """Write every cycle through the real recorder, one snapshot per reading."""
    if HISTORY.exists():
        LOG.info("history already converted: %s", HISTORY)
        return
    recorder = JsonlRecorder(HISTORY)
    features = SCHEMA.features
    for n, (row, fail) in enumerate(zip(X, failed), 1):
        state = StateVector(SCHEMA)
        for (test_id, metric), value in zip(features, row):
            state.update(test_id, metric, value)
            recorder.record(test_id, metric, state.snapshot(), None, SCHEMA.schema_hash)
        recorder.finalize(suite_passed=not fail)
        if n % 2000 == 0:
            LOG.info("  converted %d/%d units", n, len(X))
    LOG.info("history written: %s", HISTORY)


def sweep(dataset, holdout_mask, probabilities) -> tuple[float, dict]:
    """Best threshold by time saved, subject to the false-abort cap."""
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


def train() -> tuple[Path, float]:
    dataset = build_dataset([HISTORY], SCHEMA.schema_hash)
    train_mask, holdout_mask = temporal_split(dataset, HOLDOUT_FRACTION)
    # Snapshots with <3 readings carry no failure signal; training on them
    # dilutes the labels and drags calibrated probabilities toward the base
    # rate. Drop them here and gate the live engine with min_coverage=0.5.
    informative = (~np.isnan(dataset.X)).sum(axis=1) >= 3
    train_mask &= informative
    holdout_mask &= informative
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
            # isotonic keeps the top of the probability range usable; Platt
            # (sigmoid) squashes it under heavy class imbalance
            schema_hash=SCHEMA.schema_hash, calibration="isotonic", **params,
        )
        probs = predict_rows(predictor, dataset.X[holdout_mask])
        threshold, metrics = sweep(dataset, holdout_mask, probs)
        results[cls.kind] = (predictor, threshold, metrics)
        LOG.info("%-28s best t=%.3f  catch=%.0f%%  saved=%.0f%%  "
                 "false_aborts=%.1f%%  brier=%.3f",
                 cls.__name__, threshold,
                 100 * (metrics["early_catch_rate"] or 0),
                 100 * metrics["mean_fraction_saved"],
                 100 * (metrics["false_abort_rate"] or 0),
                 metrics["brier"])

    rf, gb = results["random_forest"], results["hist_gradient_boosting"]
    predictor, threshold, metrics = gb if should_promote(gb[2], rf[2]) else rf
    LOG.info("promoted: %s at threshold %.3f", type(predictor).__name__, threshold)
    artifact = ModelRegistry(REGISTRY_DIR).save(
        predictor, metrics, extra={"threshold": threshold}
    )
    return artifact, threshold


def replay_unit(engine: ValidationEngine, row) -> tuple[list[float], int | None]:
    probs: list[float] = []
    for (test_id, metric), value in zip(SCHEMA.features, row):
        try:
            p = engine.ingest(test_id, metric, float(value))
        except MLThresholdAbort as exc:
            probs.append(exc.probability)
            return probs, len(probs) - 1
        probs.append(p if p is not None else float("nan"))
    return probs, None


def replay(X: np.ndarray, failed: np.ndarray, artifact: Path, threshold: float) -> None:
    predictor = load_predictor(str(artifact))
    n_holdout = round(len(X) * HOLDOUT_FRACTION)
    holdout = np.arange(len(X))[-n_holdout:]  # rows are already in UDI order
    fail_units = [u for u in holdout if failed[u]]
    pass_units = np.random.default_rng(0).choice(
        [u for u in holdout if not failed[u]], N_PASSING_SAMPLE, replace=False
    )

    def fresh_engine() -> ValidationEngine:
        # min_coverage matches the >=3-readings training filter: no
        # predictions on warm-up states the model never saw.
        return ValidationEngine(SCHEMA, predictor, threshold=threshold,
                                min_coverage=0.5)

    caught, abort_step, aborted_pass = 0, [], 0
    for u in fail_units:
        _, abort_at = replay_unit(fresh_engine(), X[u])
        if abort_at is not None:
            caught += 1
            abort_step.append(abort_at + 1)
    for u in pass_units:
        _, abort_at = replay_unit(fresh_engine(), X[u])
        aborted_pass += abort_at is not None

    LOG.info("live replay of holdout: %d/%d failing units aborted early (%.0f%%), "
             "mean abort at reading %.1f/6",
             caught, len(fail_units), 100 * caught / len(fail_units),
             float(np.mean(abort_step)) if abort_step else float("nan"))
    LOG.info("passing units falsely aborted: %d/%d", aborted_pass, len(pass_units))

    # Determinism: identical trajectories on a repeated replay.
    unit = fail_units[0]
    p1, a1 = replay_unit(fresh_engine(), X[unit])
    p2, a2 = replay_unit(fresh_engine(), X[unit])
    same = a1 == a2 and all(
        x == y or (np.isnan(x) and np.isnan(y)) for x, y in zip(p1, p2)
    )
    LOG.info("determinism check (unit %d twice): %s", unit,
             "IDENTICAL" if same else "MISMATCH")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("predictor").setLevel(logging.WARNING)
    download()
    X, failed = load_units()
    LOG.info("AI4I 2020: %d cycles, %d failures (%.1f%%)",
             len(X), int(failed.sum()), 100 * failed.mean())
    convert(X, failed)
    artifact, threshold = train()
    replay(X, failed, artifact, threshold)


if __name__ == "__main__":
    main()
