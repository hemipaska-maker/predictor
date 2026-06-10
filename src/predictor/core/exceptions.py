"""Abort exception hierarchy raised by the orchestration engine."""

from __future__ import annotations


class OrchestrationAbortError(Exception):
    """Base class for every engine-initiated halt of a validation run."""


class MLThresholdAbort(OrchestrationAbortError):
    """Predicted suite-failure probability crossed the configured threshold."""

    def __init__(self, probability: float, threshold: float) -> None:
        self.probability = probability
        self.threshold = threshold
        super().__init__(
            f"predicted suite-failure probability {probability:.1%} "
            f">= threshold {threshold:.1%}"
        )


class RuleAbort(OrchestrationAbortError):
    """A deterministic rule fired; the ML model was bypassed entirely."""

    def __init__(self, rule_name: str, reason: str) -> None:
        self.rule_name = rule_name
        self.reason = reason
        super().__init__(f"rule '{rule_name}' triggered abort: {reason}")
