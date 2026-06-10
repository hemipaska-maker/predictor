"""Replay real SECOM holdout units through the live engine.

Proves three things on real data:
1. Known-failing units trigger MLThresholdAbort before their run completes.
2. Determinism: replaying the same unit twice gives the identical probability
   sequence and abort point.
3. Honesty: only holdout units (never seen in training) are replayed.

Run train.py first. Then:  python examples/secom/replay.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REGISTRY_DIR, build_schema, load_raw, select_sensors, unit_readings

from predictor import MLThresholdAbort, ValidationEngine
from predictor.models import load_predictor

LOG = logging.getLogger("secom.replay")

HOLDOUT_FRACTION = 0.2
N_PASSING_SAMPLE = 30  # passing units to replay (all failing units are replayed)


def replay_unit(engine: ValidationEngine, row, keep) -> tuple[list[float], int | None]:
    """Feed one unit's readings; return (probabilities, abort_reading_index)."""
    probs: list[float] = []
    for i, (test_id, metric, value) in enumerate(unit_readings(row, keep)):
        try:
            p = engine.ingest(test_id, metric, value)
        except MLThresholdAbort as exc:
            probs.append(exc.probability)
            return probs, i
        probs.append(p if p is not None else float("nan"))
    return probs, None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("predictor").setLevel(logging.ERROR)

    X, failed, ts = load_raw()
    keep = select_sensors(X)
    schema = build_schema(keep)

    # Same temporal protocol as training: the newest 20% of units are holdout.
    order = np.argsort(ts)
    n_holdout = max(1, round(len(order) * HOLDOUT_FRACTION))
    holdout = order[-n_holdout:]

    artifact = sorted(REGISTRY_DIR.glob("model_v*.joblib"))[-1]
    threshold = json.loads(
        artifact.with_suffix(".json").read_text(encoding="utf-8")
    )["threshold"]
    predictor = load_predictor(str(artifact))
    LOG.info("model: %s  threshold=%.2f  holdout=%d units", artifact.name,
             threshold, len(holdout))

    fail_units = [u for u in holdout if failed[u]]
    rng = np.random.default_rng(0)
    pass_units = rng.choice(
        [u for u in holdout if not failed[u]], N_PASSING_SAMPLE, replace=False
    )

    def fresh_engine() -> ValidationEngine:
        return ValidationEngine(schema, predictor, threshold=threshold)

    caught, total_readings, aborted_pass = 0, 0, 0
    abort_points = {}
    for unit in fail_units:
        probs, abort_at = replay_unit(fresh_engine(), X[unit], keep)
        n_readings = sum(1 for _ in unit_readings(X[unit], keep))
        total_readings += n_readings
        if abort_at is not None:
            caught += 1
            abort_points[unit] = abort_at
            LOG.info("  FAIL unit %4d: aborted at reading %d/%d (P=%.2f)",
                     unit, abort_at + 1, n_readings, probs[-1])
        else:
            LOG.info("  FAIL unit %4d: completed without abort (missed)", unit)
    for unit in pass_units:
        _, abort_at = replay_unit(fresh_engine(), X[unit], keep)
        if abort_at is not None:
            aborted_pass += 1
            LOG.info("  PASS unit %4d: falsely aborted at reading %d", unit, abort_at + 1)

    LOG.info("")
    LOG.info("failing units caught early: %d/%d (%.0f%%)",
             caught, len(fail_units), 100 * caught / len(fail_units))
    LOG.info("passing units falsely aborted: %d/%d",
             aborted_pass, len(pass_units))
    if abort_points:
        mean_frac = np.mean([
            abort_points[u] / sum(1 for _ in unit_readings(X[u], keep))
            for u in abort_points
        ])
        LOG.info("caught failures aborted after %.0f%% of their readings "
                 "on average", 100 * mean_frac)

    # Determinism: same unit, two fresh engines -> identical trajectories.
    if abort_points:
        unit = next(iter(abort_points))
        p1, a1 = replay_unit(fresh_engine(), X[unit], keep)
        p2, a2 = replay_unit(fresh_engine(), X[unit], keep)
        identical = a1 == a2 and all(
            x == y or (np.isnan(x) and np.isnan(y)) for x, y in zip(p1, p2)
        )
        LOG.info("determinism check (unit %d replayed twice): %s",
                 unit, "IDENTICAL" if identical else "MISMATCH")


if __name__ == "__main__":
    main()
