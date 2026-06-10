"""MLOps pipeline: dataset building, retraining, evaluation, drift, registry.

See DESIGN.md §4. Requires the [sklearn] extra.
"""

from .dataset import Dataset, build_dataset, iter_runs, temporal_split
from .drift import drift_report, population_stability_index
from .evaluate import brier_score, evaluate, predict_rows, simulate_fail_fast
from .registry import ModelRegistry
from .train import TrainResult, should_promote, train_candidate

__all__ = [
    "Dataset",
    "ModelRegistry",
    "TrainResult",
    "brier_score",
    "build_dataset",
    "drift_report",
    "evaluate",
    "iter_runs",
    "population_stability_index",
    "predict_rows",
    "should_promote",
    "simulate_fail_fast",
    "temporal_split",
    "train_candidate",
]
