"""Retraining pipeline entry point (DESIGN.md §4.2 steps 2-4).

Meant to run on a schedule (cron/CI):

    from predictor.models import RandomForestPredictor
    from predictor.training import build_dataset, train_candidate

    ds = build_dataset(glob("records/*.jsonl"), schema.schema_hash)
    result = train_candidate(ds, RandomForestPredictor, threshold=0.85)
    if should_promote(result.metrics, current_metrics):
        registry.save(result.predictor, result.metrics)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dataset import Dataset, temporal_split
from .evaluate import evaluate

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    predictor: Any                 # SklearnPredictor, ready to save/deploy
    metrics: dict                  # holdout report from evaluate()
    n_train_runs: int
    n_holdout_runs: int
    params: dict = field(default_factory=dict)


def train_candidate(
    dataset: Dataset,
    model_cls: type,
    *,
    threshold: float = 0.85,
    holdout_fraction: float = 0.2,
    calibration: str | None = "isotonic",
    **estimator_params: Any,
) -> TrainResult:
    """Fit + calibrate a candidate on a temporal split, evaluate on holdout.

    ``model_cls`` is an adapter class (RandomForestPredictor or
    GradientBoostingPredictor); its ``train`` owns the NaN transform so
    training and inference can never diverge.
    """
    train_mask, holdout_mask = temporal_split(dataset, holdout_fraction)
    predictor = model_cls.train(
        dataset.X[train_mask],
        dataset.y[train_mask],
        schema_hash=dataset.schema_hash,
        calibration=calibration,
        **estimator_params,
    )
    metrics = evaluate(predictor, dataset, holdout_mask, threshold)
    logger.info(
        "candidate %s: catch=%s saved=%s false_aborts=%s brier=%.4f",
        model_cls.__name__,
        metrics["early_catch_rate"],
        metrics["mean_fraction_saved"],
        metrics["false_abort_rate"],
        metrics["brier"],
    )
    return TrainResult(
        predictor=predictor,
        metrics=metrics,
        n_train_runs=len(np.unique(dataset.run_id[train_mask])),
        n_holdout_runs=len(np.unique(dataset.run_id[holdout_mask])),
        params={"calibration": calibration, **estimator_params},
    )


def should_promote(
    candidate: dict,
    incumbent: dict | None,
    *,
    max_false_abort_rate: float = 0.05,
) -> bool:
    """Promotion gate on the business metric, not raw accuracy.

    A candidate must keep false aborts under the cap (killing good runs
    erodes trust fastest), then beat the incumbent on time saved; Brier
    breaks ties.
    """
    far = candidate.get("false_abort_rate")
    if far is not None and far > max_false_abort_rate:
        return False
    if incumbent is None:
        return True
    cand_saved = candidate.get("mean_fraction_saved") or 0.0
    inc_saved = incumbent.get("mean_fraction_saved") or 0.0
    if cand_saved != inc_saved:
        return cand_saved > inc_saved
    return candidate.get("brier", 1.0) < incumbent.get("brier", 1.0)
