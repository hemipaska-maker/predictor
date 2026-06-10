"""Download the UCI SECOM dataset (CC BY 4.0) into the local data dir."""

from __future__ import annotations

import hashlib
import logging
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RAW_DATA, RAW_LABELS

LOG = logging.getLogger("secom.download")

BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom"
FILES = {RAW_DATA: f"{BASE}/secom.data", RAW_LABELS: f"{BASE}/secom_labels.data"}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in FILES.items():
        if path.exists():
            LOG.info("already present: %s", path)
            continue
        LOG.info("downloading %s ...", url)
        urllib.request.urlretrieve(url, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        LOG.info("  -> %s (%d bytes, sha256=%s)", path, path.stat().st_size, digest[:16])
    LOG.info("done. data dir: %s", DATA_DIR)


if __name__ == "__main__":
    main()
