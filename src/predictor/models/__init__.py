"""Sklearn-backed strategies. Requires `pip install predictor[sklearn]`.

See DESIGN.md §3.2. Import errors here mean the extra is not installed —
the core never imports this package.
"""

from .sklearn_models import (
    GradientBoostingPredictor,
    RandomForestPredictor,
    SklearnPredictor,
    load_predictor,
)

__all__ = [
    "GradientBoostingPredictor",
    "RandomForestPredictor",
    "SklearnPredictor",
    "load_predictor",
]
