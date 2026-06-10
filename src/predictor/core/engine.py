"""The orchestration core: synchronous ingestion, rules, prediction, abort."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from .exceptions import MLThresholdAbort, RuleAbort
from .records import RunRecorder
from .rules import Rule
from .state import FeatureSchema, StateVector
from .strategy import FailurePredictor

logger = logging.getLogger(__name__)


class ValidationEngine:
    """Single-threaded fail-fast engine.

    Tests push measurements through :meth:`ingest`; the engine updates the
    fixed state vector, consults deterministic rules and then the predictive
    strategy, and raises :class:`OrchestrationAbortError` subclasses to halt
    the run. It never touches hardware — teardown belongs to the host runner.
    """

    def __init__(
        self,
        schema: FeatureSchema,
        predictor: FailurePredictor,
        *,
        threshold: float = 0.85,
        rules: Iterable[Rule] = (),
        min_coverage: float = 0.0,
        recorder: RunRecorder | None = None,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0.0, 1.0]")
        if not 0.0 <= min_coverage <= 1.0:
            raise ValueError("min_coverage must be in [0.0, 1.0]")
        # A model trained against a different feature layout would read the
        # wrong indices and predict garbage silently — refuse up front.
        if (
            predictor.schema_hash is not None
            and predictor.schema_hash != schema.schema_hash
        ):
            raise ValueError(
                "predictor was trained on a different feature schema "
                f"({predictor.schema_hash[:8]} != {schema.schema_hash[:8]}); "
                "retrain or fix the schema before running"
            )
        self._schema = schema
        self._predictor = predictor
        self._threshold = threshold
        self._rules = tuple(rules)
        self._min_coverage = min_coverage
        self._recorder = recorder
        self._state = StateVector(schema)
        self._last_probability: float | None = None
        logger.debug(
            "engine ready: %d features, threshold=%.0f%%, min_coverage=%.0f%%, "
            "%d rules, predictor=%s",
            len(schema),
            threshold * 100,
            min_coverage * 100,
            len(self._rules),
            type(predictor).__name__,
        )

    @property
    def state(self) -> StateVector:
        return self._state

    @property
    def last_probability(self) -> float | None:
        return self._last_probability

    def ingest(self, test_id: str, metric: str, value: float) -> float | None:
        """Feed one measurement; return P(suite failure) or None during warm-up.

        Raises:
            RuleAbort: a deterministic safety rule fired (ML bypassed).
            MLThresholdAbort: predicted probability crossed the threshold.
        """
        # Safety rules run before anything else — a catastrophic reading must
        # abort even if it would also crash the model or corrupt the state.
        for rule in self._rules:
            reason = rule.check(test_id, metric, value)
            if reason is not None:
                logger.warning("rule %s fired: %s", rule.name, reason)
                raise RuleAbort(rule.name, reason)

        self._state.update(test_id, metric, value)

        probability: float | None = None
        if self._state.coverage() >= self._min_coverage:
            probability = self._predictor.predict_failure_probability(
                self._state.snapshot()
            )
            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"predictor returned {probability!r}, outside [0, 1]"
                )
            self._last_probability = probability
        logger.debug(
            "%s.%s=%.6g -> P(fail)=%s",
            test_id,
            metric,
            value,
            "warm-up" if probability is None else f"{probability:.3f}",
        )

        # Record before deciding: the snapshot that triggers an abort is
        # exactly the kind of row the next retraining run needs to see.
        if self._recorder is not None:
            self._recorder.record(
                test_id,
                metric,
                self._state.snapshot(),
                probability,
                self._schema.schema_hash,
            )

        if probability is not None and probability >= self._threshold:
            logger.warning(
                "ML abort: P(fail)=%.3f >= threshold %.3f after %s.%s",
                probability,
                self._threshold,
                test_id,
                metric,
            )
            raise MLThresholdAbort(probability, self._threshold)
        return probability

    def finalize(self, suite_passed: bool) -> None:
        """Label the run for training data; call from the runner's teardown."""
        logger.info("run finalized: suite_passed=%s", suite_passed)
        if self._recorder is not None:
            self._recorder.finalize(suite_passed)
