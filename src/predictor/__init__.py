"""ML-driven hardware validation engine with a fail-fast core."""

import logging

from .core import (
    BoundsRule,
    FailurePredictor,
    FeatureSchema,
    JsonlRecorder,
    MLThresholdAbort,
    OrchestrationAbortError,
    Rule,
    RuleAbort,
    StateVector,
    ValidationEngine,
)

__version__ = "0.1.0.dev0"

# Library convention: emit nothing unless the application configures logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "BoundsRule",
    "FailurePredictor",
    "FeatureSchema",
    "JsonlRecorder",
    "MLThresholdAbort",
    "OrchestrationAbortError",
    "Rule",
    "RuleAbort",
    "StateVector",
    "ValidationEngine",
]
