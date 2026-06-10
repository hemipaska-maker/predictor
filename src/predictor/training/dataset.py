"""Build training datasets from JSONL run records (DESIGN.md §4.2 step 1).

Every intermediate state snapshot becomes a training row labeled with the
run's *final* outcome — this is what teaches the model to predict from
partial, NaN-heavy vectors. Runs are kept identifiable so splits can be
temporal and per-run (random row splits would leak a run's own future
snapshots into training).
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecordedRun:
    """One finalized suite run reconstructed from JSONL events."""

    snapshots: list[list[float]]   # progressive state vectors, in order
    failed: bool                   # the label every snapshot inherits
    schema_hash: str
    start_ts: float


@dataclass(frozen=True)
class Dataset:
    """Rows of state snapshots with run-level provenance."""

    X: np.ndarray          # (n_rows, n_features) with NaN placeholders
    y: np.ndarray          # (n_rows,) 1 = suite failed, 0 = passed
    run_id: np.ndarray     # (n_rows,) integer run index, ordered by start time
    schema_hash: str

    @property
    def n_runs(self) -> int:
        return len(np.unique(self.run_id))

    def rows_for_runs(self, runs: np.ndarray) -> np.ndarray:
        """Boolean row mask selecting the given run ids."""
        return np.isin(self.run_id, runs)


def iter_runs(paths: list[str | Path]) -> list[RecordedRun]:
    """Parse JSONL files into finalized runs.

    A run is a sequence of 'ingest' events terminated by a 'finalize' event.
    Unfinalized trailing runs (crashed sessions never labeled) are dropped
    with a warning — an abort still gets finalized by the host runner, so a
    missing label means the outcome is genuinely unknown.
    """
    runs: list[RecordedRun] = []
    for path in paths:
        snapshots: list[list[float]] = []
        hashes: set[str] = set()
        start_ts: float | None = None
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if event["event"] == "ingest":
                    if start_ts is None:
                        start_ts = event["ts"]
                    hashes.add(event["schema_hash"])
                    # JSON has no NaN; the recorder wrote placeholders as null.
                    snapshots.append(
                        [np.nan if v is None else v for v in event["state"]]
                    )
                elif event["event"] == "finalize":
                    if snapshots:
                        if len(hashes) != 1:
                            raise ValueError(
                                f"{path}: run mixes schema hashes {hashes}"
                            )
                        runs.append(
                            RecordedRun(
                                snapshots=snapshots,
                                failed=not event["suite_passed"],
                                schema_hash=hashes.pop(),
                                start_ts=start_ts,
                            )
                        )
                    snapshots, hashes, start_ts = [], set(), None
        if snapshots:
            warnings.warn(
                f"{path}: dropped unfinalized run ({len(snapshots)} snapshots)",
                stacklevel=2,
            )
    logger.debug("parsed %d finalized runs from %d files", len(runs), len(paths))
    return runs


def build_dataset(paths: list[str | Path], schema_hash: str) -> Dataset:
    """Assemble rows from all runs matching ``schema_hash``, oldest first."""
    matching = [r for r in iter_runs(paths) if r.schema_hash == schema_hash]
    if not matching:
        raise ValueError(f"no finalized runs found for schema {schema_hash[:8]}")
    matching.sort(key=lambda r: r.start_ts)

    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    run_rows: list[int] = []
    for run_idx, run in enumerate(matching):
        for snap in run.snapshots:
            X_rows.append(snap)
            y_rows.append(int(run.failed))
            run_rows.append(run_idx)
    logger.info(
        "dataset for schema %s: %d rows from %d runs (%d failed)",
        schema_hash[:8],
        len(X_rows),
        len(matching),
        sum(r.failed for r in matching),
    )
    return Dataset(
        X=np.asarray(X_rows, dtype=float),
        y=np.asarray(y_rows, dtype=int),
        run_id=np.asarray(run_rows, dtype=int),
        schema_hash=schema_hash,
    )


def temporal_split(
    dataset: Dataset, holdout_fraction: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Split into (train_mask, holdout_mask) by run recency.

    The newest runs form the holdout — never a random split, which would
    leak future hardware behavior into the past (DESIGN.md §4.2).
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    runs = np.unique(dataset.run_id)  # already ordered oldest-first
    n_holdout = max(1, round(len(runs) * holdout_fraction))
    if n_holdout >= len(runs):
        raise ValueError(f"need at least 2 runs to split; have {len(runs)}")
    return (
        dataset.rows_for_runs(runs[:-n_holdout]),
        dataset.rows_for_runs(runs[-n_holdout:]),
    )
