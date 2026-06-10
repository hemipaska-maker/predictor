"""Strategy interface: the only contract the engine has with any ML model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class FailurePredictor(ABC):
    """Interchangeable predictive core (Strategy pattern).

    Implementations must be deterministic, stateless functions of the input
    vector so that replaying a run log reproduces identical decisions. They
    return a probability — the abort decision belongs to the engine alone.
    """

    @abstractmethod
    def predict_failure_probability(self, state: Sequence[float]) -> float:
        """Return P(overall suite failure) in [0.0, 1.0].

        ``state`` follows the engine's FeatureSchema layout; unobserved
        features are NaN. Adapters that cannot consume NaN must impute here.
        """

    @property
    def schema_hash(self) -> str | None:
        """Hash of the schema the model was trained on, or None if untracked.

        The engine rejects predictors whose hash mismatches the live schema.
        """
        return None
