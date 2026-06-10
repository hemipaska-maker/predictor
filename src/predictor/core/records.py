"""Run-record capture: the raw material for retraining pipelines."""

from __future__ import annotations

import json
import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)


class RunRecorder(ABC):
    """Sink for prediction events and the final suite outcome."""

    @abstractmethod
    def record(
        self,
        test_id: str,
        metric: str,
        state: Sequence[float],
        probability: float | None,
        schema_hash: str,
    ) -> None:
        """Log one ingestion event (probability is None during warm-up)."""

    @abstractmethod
    def finalize(self, suite_passed: bool) -> None:
        """Label the whole run; called by the host runner at session end."""


class JsonlRecorder(RunRecorder):
    """Append-only JSONL recorder — diffable, greppable, mergeable."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, payload: dict) -> None:
        payload["ts"] = time.time()
        # Open per write: bench processes can crash at any point, and an
        # unbuffered append loses at most the event in flight.
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")

    def record(
        self,
        test_id: str,
        metric: str,
        state: Sequence[float],
        probability: float | None,
        schema_hash: str,
    ) -> None:
        self._append(
            {
                "event": "ingest",
                "test_id": test_id,
                "metric": metric,
                # NaN is not valid JSON; encode placeholders as null.
                "state": [None if math.isnan(v) else v for v in state],
                "probability": probability,
                "schema_hash": schema_hash,
            }
        )

    def finalize(self, suite_passed: bool) -> None:
        logger.debug("labeling run in %s: suite_passed=%s", self._path, suite_passed)
        self._append({"event": "finalize", "suite_passed": suite_passed})
