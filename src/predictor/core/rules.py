"""Deterministic safety rules, checked before the ML model on every value.

Rules protect hardware; the ML model saves time. A rule firing is a hard
engineering fact and always bypasses the model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Rule(ABC):
    """A hardcoded check applied to every ingested measurement."""

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def check(self, test_id: str, metric: str, value: float) -> str | None:
        """Return a human-readable violation reason, or None if the value is safe."""


class BoundsRule(Rule):
    """Abort when a named metric leaves its safe range (any test_id)."""

    def __init__(
        self,
        metric: str,
        *,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        if min_value is None and max_value is None:
            raise ValueError("BoundsRule needs min_value and/or max_value")
        self.metric = metric
        self.min_value = min_value
        self.max_value = max_value

    @property
    def name(self) -> str:
        return f"BoundsRule({self.metric})"

    def check(self, test_id: str, metric: str, value: float) -> str | None:
        if metric != self.metric:
            return None
        if self.min_value is not None and value < self.min_value:
            return (
                f"{metric}={value} from {test_id} below safe minimum "
                f"{self.min_value}"
            )
        if self.max_value is not None and value > self.max_value:
            return (
                f"{metric}={value} from {test_id} above safe maximum "
                f"{self.max_value}"
            )
        return None
