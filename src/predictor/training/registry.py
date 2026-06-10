"""Versioned model artifact registry (DESIGN.md §4.2 step 4).

Layout inside the registry directory:

    model_v3_1a2b3c4d.joblib        # adapter artifact (joblib)
    model_v3_1a2b3c4d.json          # metadata sidecar: metrics, params, window

Versions are monotonic per schema hash; `latest()` is what deployments load.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_NAME = re.compile(r"^model_v(?P<version>\d+)_(?P<hash8>[0-9a-f]{8})\.joblib$")


class ModelRegistry:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _entries(self, schema_hash: str) -> list[tuple[int, Path]]:
        hash8 = schema_hash[:8]
        out = []
        for path in self._root.glob("model_v*.joblib"):
            m = _NAME.match(path.name)
            if m and m.group("hash8") == hash8:
                out.append((int(m.group("version")), path))
        return sorted(out)

    def save(self, predictor, metrics: dict, extra: dict | None = None) -> Path:
        """Persist a promoted model as the next version for its schema."""
        if predictor.schema_hash is None:
            raise ValueError("predictor has no schema_hash; refuse to register")
        entries = self._entries(predictor.schema_hash)
        version = entries[-1][0] + 1 if entries else 1
        stem = f"model_v{version}_{predictor.schema_hash[:8]}"
        artifact = self._root / f"{stem}.joblib"
        predictor.save(str(artifact))
        sidecar = {
            "version": version,
            "kind": predictor.kind,
            "schema_hash": predictor.schema_hash,
            "metrics": metrics,
            "metadata": predictor.metadata,
        }
        sidecar.update(extra or {})
        (self._root / f"{stem}.json").write_text(
            json.dumps(sidecar, indent=2), encoding="utf-8"
        )
        logger.info("registered %s (kind=%s)", artifact.name, predictor.kind)
        return artifact

    def latest(self, schema_hash: str) -> Path | None:
        """Path of the newest artifact for this schema, or None."""
        entries = self._entries(schema_hash)
        return entries[-1][1] if entries else None

    def latest_metrics(self, schema_hash: str) -> dict | None:
        """Holdout metrics of the incumbent — feed to should_promote()."""
        path = self.latest(schema_hash)
        if path is None:
            return None
        sidecar = path.with_suffix(".json")
        if not sidecar.exists():
            return None
        return json.loads(sidecar.read_text(encoding="utf-8")).get("metrics")
