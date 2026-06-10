"""Model evaluation (DESIGN.md §4.2 step 3).

Two views matter:
- Brier score / calibration — threshold semantics depend on probabilities
  meaning what they say.
- The business metric — a replay of the fail-fast policy over held-out runs:
  machine time saved on truly failing runs vs. good runs wrongly aborted.
"""

from __future__ import annotations

import numpy as np

from predictor.core.strategy import FailurePredictor

from .dataset import Dataset


def brier_score(probabilities: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error of probabilities vs. outcomes; lower is better."""
    p = np.asarray(probabilities, dtype=float)
    return float(np.mean((p - np.asarray(y, dtype=float)) ** 2))


def predict_rows(predictor: FailurePredictor, X: np.ndarray) -> np.ndarray:
    """Run the strategy row by row, exactly as the engine would."""
    return np.asarray(
        [predictor.predict_failure_probability(row) for row in X], dtype=float
    )


def simulate_fail_fast(
    dataset: Dataset,
    mask: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    """Replay the abort policy over each selected run, in snapshot order.

    ``probabilities`` is aligned with the full dataset rows (values outside
    ``mask`` are ignored). Returns the business metrics:
    - early_catch_rate: failing runs aborted before completion.
    - mean_fraction_saved: mean fraction of snapshots skipped across ALL
      failing runs — a missed run saves 0. (Averaging only caught runs would
      reward a model that catches one run early and misses the rest.)
    - false_abort_rate: passing runs wrongly killed — the cost side.
    """
    caught, saved_fractions, false_aborts = 0, [], 0
    n_failed_runs, n_passed_runs = 0, 0

    for run in np.unique(dataset.run_id[mask]):
        rows = mask & (dataset.run_id == run)
        probs = probabilities[rows]
        failed = bool(dataset.y[rows][0])
        n_steps = int(rows.sum())
        crossed = np.flatnonzero(probs >= threshold)
        aborted_at = int(crossed[0]) if crossed.size else None

        if failed:
            n_failed_runs += 1
            if aborted_at is not None:
                caught += 1
                saved_fractions.append(1.0 - (aborted_at + 1) / n_steps)
            else:
                saved_fractions.append(0.0)
        else:
            n_passed_runs += 1
            if aborted_at is not None:
                false_aborts += 1

    return {
        "threshold": threshold,
        "n_failed_runs": n_failed_runs,
        "n_passed_runs": n_passed_runs,
        "early_catch_rate": caught / n_failed_runs if n_failed_runs else None,
        "mean_fraction_saved": (
            float(np.mean(saved_fractions)) if saved_fractions else 0.0
        ),
        "false_abort_rate": (
            false_aborts / n_passed_runs if n_passed_runs else None
        ),
    }


def evaluate(
    predictor: FailurePredictor,
    dataset: Dataset,
    mask: np.ndarray,
    threshold: float,
) -> dict:
    """Full holdout report: calibration quality + replayed business metrics."""
    probabilities = predict_rows(predictor, dataset.X[mask])
    full_probs = np.zeros(len(dataset.y))
    full_probs[mask] = probabilities
    report = simulate_fail_fast(dataset, mask, full_probs, threshold)
    report["brier"] = brier_score(probabilities, dataset.y[mask])
    report["n_rows"] = int(mask.sum())
    return report
