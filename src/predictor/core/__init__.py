"""Framework-agnostic core — stdlib only, zero external dependencies."""

from .engine import ValidationEngine
from .exceptions import MLThresholdAbort, OrchestrationAbortError, RuleAbort
from .records import JsonlRecorder, RunRecorder
from .rules import BoundsRule, Rule
from .state import PLACEHOLDER, FeatureSchema, StateVector
from .strategy import FailurePredictor

__all__ = [
    "PLACEHOLDER",
    "BoundsRule",
    "FailurePredictor",
    "FeatureSchema",
    "JsonlRecorder",
    "MLThresholdAbort",
    "OrchestrationAbortError",
    "Rule",
    "RuleAbort",
    "RunRecorder",
    "StateVector",
    "ValidationEngine",
]
