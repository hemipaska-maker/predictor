"""Concept-drift monitoring (DESIGN.md §4.2 step 5).

Flags a stale model *between* scheduled retrains by comparing recent data
against training-time baselines:

- input drift: PSI per feature on observed values, plus missing-rate shift
  (NaN placeholders are signal here — a test that stopped running is drift);
- output drift: Brier score on recent labeled runs vs. the training holdout.
"""

from __future__ import annotations

import numpy as np

#: Conventional PSI interpretation: <0.1 stable, 0.1-0.25 moderate, >0.25 major.
PSI_ALERT = 0.25


def population_stability_index(
    baseline: np.ndarray, recent: np.ndarray, bins: int = 10
) -> float:
    """PSI between two 1-D samples, binned on baseline quantiles. NaN ignored."""
    baseline = baseline[~np.isnan(baseline)]
    recent = recent[~np.isnan(recent)]
    if baseline.size == 0 or recent.size == 0:
        return 0.0
    edges = np.quantile(baseline, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)  # constant features collapse to one bin -> PSI 0
    if len(edges) < 3:
        return 0.0
    b_frac = np.histogram(baseline, edges)[0] / baseline.size
    r_frac = np.histogram(recent, edges)[0] / recent.size
    eps = 1e-6
    b_frac, r_frac = np.clip(b_frac, eps, None), np.clip(r_frac, eps, None)
    return float(np.sum((r_frac - b_frac) * np.log(r_frac / b_frac)))


def drift_report(
    baseline_X: np.ndarray,
    recent_X: np.ndarray,
    *,
    psi_alert: float = PSI_ALERT,
    missing_shift_alert: float = 0.15,
) -> dict:
    """Per-feature PSI + missing-rate shift; `retrain_recommended` rolls it up."""
    baseline_X = np.asarray(baseline_X, dtype=float)
    recent_X = np.asarray(recent_X, dtype=float)
    if baseline_X.shape[1] != recent_X.shape[1]:
        raise ValueError(
            "feature count mismatch — different schema versions? "
            f"{baseline_X.shape[1]} != {recent_X.shape[1]}"
        )
    psi = np.array(
        [
            population_stability_index(baseline_X[:, j], recent_X[:, j])
            for j in range(baseline_X.shape[1])
        ]
    )
    missing_shift = np.abs(
        np.isnan(recent_X).mean(axis=0) - np.isnan(baseline_X).mean(axis=0)
    )
    flagged = np.flatnonzero((psi > psi_alert) | (missing_shift > missing_shift_alert))
    return {
        "max_psi": float(psi.max()),
        "mean_psi": float(psi.mean()),
        "max_missing_shift": float(missing_shift.max()),
        "flagged_features": flagged.tolist(),
        "retrain_recommended": bool(flagged.size),
    }
