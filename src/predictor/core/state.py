"""Fixed-vector state management: feature schema and the in-memory state."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

#: Canonical placeholder meaning "test not yet run". Model adapters decide
#: how to handle it (native NaN support, imputation, etc.).
PLACEHOLDER = float("nan")


class FeatureSchema:
    """Ordered, immutable layout of (test_id, metric) pairs.

    The order defines the model's input vector. Any change to the layout
    changes ``schema_hash``, which invalidates previously trained models.
    """

    def __init__(self, features: Sequence[tuple[str, str]]) -> None:
        if not features:
            raise ValueError("schema must declare at least one feature")
        self._features: tuple[tuple[str, str], ...] = tuple(
            (str(t), str(m)) for t, m in features
        )
        self._index: dict[tuple[str, str], int] = {}
        for i, key in enumerate(self._features):
            if key in self._index:
                raise ValueError(f"duplicate feature {key!r}")
            self._index[key] = i

    def __len__(self) -> int:
        return len(self._features)

    @property
    def features(self) -> tuple[tuple[str, str], ...]:
        return self._features

    def index_of(self, test_id: str, metric: str) -> int:
        try:
            return self._index[(test_id, metric)]
        except KeyError:
            raise KeyError(
                f"({test_id!r}, {metric!r}) is not declared in the schema"
            ) from None

    @property
    def schema_hash(self) -> str:
        """SHA-256 over the canonical layout; stored in every model artifact."""
        canonical = "\n".join(f"{t}\x1f{m}" for t, m in self._features)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class StateVector:
    """Fixed-width vector of observed metric values, NaN where not yet run."""

    def __init__(self, schema: FeatureSchema) -> None:
        self._schema = schema
        self._values: list[float] = [PLACEHOLDER] * len(schema)
        self._observed = 0

    @property
    def schema(self) -> FeatureSchema:
        return self._schema

    def update(self, test_id: str, metric: str, value: float) -> int:
        """Write ``value`` at the feature's index; return the index."""
        idx = self._schema.index_of(test_id, metric)
        if math.isnan(self._values[idx]):
            self._observed += 1
        self._values[idx] = float(value)
        return idx

    def snapshot(self) -> list[float]:
        """Copy of the current vector, safe to hand to a model or recorder."""
        return list(self._values)

    def coverage(self) -> float:
        """Fraction of features observed so far, in [0.0, 1.0]."""
        return self._observed / len(self._values)

    def reset(self) -> None:
        self._values = [PLACEHOLDER] * len(self._schema)
        self._observed = 0
