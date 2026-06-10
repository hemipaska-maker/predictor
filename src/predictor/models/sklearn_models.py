"""Sklearn-backed FailurePredictor adapters (DESIGN.md §3.2).

Both adapters own their NaN handling: the core's placeholder semantics never
leak model-specific sentinels. Trained models are calibrated so user
thresholds carry real probability meaning, and every artifact stores the
schema hash it was trained against.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, ClassVar

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from predictor.core.strategy import FailurePredictor

logger = logging.getLogger(__name__)

_ARTIFACT_FORMAT = 1


class SklearnPredictor(FailurePredictor):
    """Common adapter machinery: transform, predict, persist."""

    #: Subclasses set a stable identifier stored in artifacts.
    kind: ClassVar[str] = ""

    def __init__(
        self,
        model: Any,
        *,
        schema_hash: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._model = model
        self._schema_hash = schema_hash
        self.metadata = dict(metadata or {})
        classes = list(getattr(model, "classes_", []))
        if 1 not in classes:
            raise ValueError(
                "model must be fitted on binary labels with 1 = suite failed; "
                f"got classes {classes!r}"
            )
        self._positive_idx = classes.index(1)

    @property
    def schema_hash(self) -> str | None:
        return self._schema_hash

    # -- NaN handling -----------------------------------------------------

    @classmethod
    def transform(cls, X: np.ndarray) -> np.ndarray:
        """Map raw state matrix (with NaN placeholders) to model features.

        Must be applied identically at training and inference time; both
        :meth:`train` and :meth:`predict_failure_probability` go through it.
        """
        raise NotImplementedError

    # -- inference ----------------------------------------------------------

    def predict_failure_probability(self, state: Sequence[float]) -> float:
        X = self.transform(np.asarray([state], dtype=float))
        proba = self._model.predict_proba(X)[0, self._positive_idx]
        return float(proba)

    # -- training -----------------------------------------------------------

    @classmethod
    def default_estimator(cls, **params: Any) -> Any:
        raise NotImplementedError

    @classmethod
    def train(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        *,
        schema_hash: str | None = None,
        calibration: str | None = "isotonic",
        metadata: dict | None = None,
        **estimator_params: Any,
    ) -> "SklearnPredictor":
        """Fit (and calibrate) on rows of state snapshots labeled 0/1.

        ``calibration`` is 'isotonic', 'sigmoid', or None (raw probabilities;
        only for debugging — thresholds lose meaning without calibration).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError(
                "training labels must contain both classes 0 (passed) and "
                f"1 (failed); got {sorted(np.unique(y))}"
            )
        logger.info(
            "training %s on %d rows (%d failures), calibration=%s",
            cls.kind,
            len(y),
            int(y.sum()),
            calibration,
        )
        estimator = cls.default_estimator(**estimator_params)
        if calibration is not None:
            estimator = CalibratedClassifierCV(estimator, method=calibration, cv=3)
        estimator.fit(cls.transform(X), y)
        meta = {"calibration": calibration, "n_train_rows": int(len(y))}
        meta.update(metadata or {})
        return cls(estimator, schema_hash=schema_hash, metadata=meta)

    # -- persistence ----------------------------------------------------------

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "format": _ARTIFACT_FORMAT,
                "kind": self.kind,
                "model": self._model,
                "schema_hash": self._schema_hash,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "SklearnPredictor":
        payload = joblib.load(path)
        kind = payload.get("kind")
        impl = _KINDS.get(kind)
        if impl is None:
            raise ValueError(f"unknown model kind {kind!r} in {path}")
        if cls is not SklearnPredictor and impl is not cls:
            raise ValueError(
                f"{path} contains a {kind!r} model, not {cls.kind!r}"
            )
        logger.debug("loaded %s model from %s", kind, path)
        return impl(
            payload["model"],
            schema_hash=payload["schema_hash"],
            metadata=payload["metadata"],
        )


class RandomForestPredictor(SklearnPredictor):
    """Stable default: robust to hardware noise, low variance.

    Random forests cannot consume NaN, so the transform imputes placeholders
    to 0 and appends a binary observed-mask per feature — the model learns
    "not yet run" as signal instead of mistaking the fill value for a reading.
    """

    kind = "random_forest"

    @classmethod
    def transform(cls, X: np.ndarray) -> np.ndarray:
        mask = np.isnan(X)
        imputed = np.where(mask, 0.0, X)
        return np.hstack([imputed, (~mask).astype(float)])

    @classmethod
    def default_estimator(cls, **params: Any) -> RandomForestClassifier:
        defaults: dict[str, Any] = {
            "n_estimators": 300,
            "min_samples_leaf": 5,
            "n_jobs": -1,
            "random_state": 0,
        }
        defaults.update(params)
        return RandomForestClassifier(**defaults)


class GradientBoostingPredictor(SklearnPredictor):
    """Higher accuracy ceiling; strictly regularized to resist overfitting.

    HistGradientBoostingClassifier treats NaN natively as "missing", which
    matches the core's placeholder semantics exactly — no transform needed.
    """

    kind = "hist_gradient_boosting"

    @classmethod
    def transform(cls, X: np.ndarray) -> np.ndarray:
        return X

    @classmethod
    def default_estimator(cls, **params: Any) -> HistGradientBoostingClassifier:
        defaults: dict[str, Any] = {
            "max_depth": 4,
            "learning_rate": 0.05,
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "l2_regularization": 1.0,
            "random_state": 0,
        }
        defaults.update(params)
        return HistGradientBoostingClassifier(**defaults)


_KINDS: dict[str, type[SklearnPredictor]] = {
    RandomForestPredictor.kind: RandomForestPredictor,
    GradientBoostingPredictor.kind: GradientBoostingPredictor,
}


def load_predictor(path: str) -> SklearnPredictor:
    """Load any saved adapter, dispatching on the artifact's `kind`."""
    return SklearnPredictor.load(path)
