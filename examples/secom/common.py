"""Shared plumbing for the SECOM real-data example.

UCI SECOM: 1,567 semiconductor production units, 590 process sensors each,
pass/fail label + timestamp per unit. We map each unit to a suite run and
group sensors (in column order) into pseudo-test "stations".

Data lives in a local temp dir by default (override with SECOM_DATA_DIR) so
large generated files stay out of the repo and cloud-synced folders.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

# Allow running from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from predictor import FeatureSchema

DATA_DIR = Path(
    os.environ.get("SECOM_DATA_DIR", Path(tempfile.gettempdir()) / "secom_predictor")
)
RAW_DATA = DATA_DIR / "secom.data"
RAW_LABELS = DATA_DIR / "secom_labels.data"
HISTORY = DATA_DIR / "history.jsonl"
REGISTRY_DIR = DATA_DIR / "registry"

SENSORS_PER_STATION = 20
MAX_MISSING_FRACTION = 0.4  # drop sensors that are mostly absent


def load_raw() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X[1567, 590], failed[1567] bool, start_ts[1567] epoch seconds)."""
    X = np.genfromtxt(RAW_DATA)  # 'NaN' tokens parse to nan
    failed, ts = [], []
    for line in RAW_LABELS.read_text().splitlines():
        if not line.strip():
            continue
        label, stamp = line.split(maxsplit=1)
        failed.append(label.strip() == "1")  # SECOM: -1 = pass, 1 = fail
        ts.append(
            datetime.strptime(stamp.strip().strip('"'), "%d/%m/%Y %H:%M:%S").timestamp()
        )
    return X, np.asarray(failed), np.asarray(ts)


def select_sensors(X: np.ndarray) -> np.ndarray:
    """Indices of usable sensors: not mostly missing, not constant.

    Unsupervised filtering (no labels involved), applied identically by
    train and replay so both sides derive the same schema.
    """
    missing = np.isnan(X).mean(axis=0)
    with np.errstate(invalid="ignore"):
        spread = np.nanstd(X, axis=0)
    keep = (missing <= MAX_MISSING_FRACTION) & (spread > 0)
    return np.flatnonzero(keep)


def build_schema(keep: np.ndarray) -> FeatureSchema:
    """Group kept sensors, in column order, into stations of 20."""
    return FeatureSchema(
        [
            (f"station_{pos // SENSORS_PER_STATION + 1:02d}", f"s{sensor:03d}")
            for pos, sensor in enumerate(keep)
        ]
    )


def unit_readings(row: np.ndarray, keep: np.ndarray):
    """Yield (test_id, metric, value) for one unit, skipping missing sensors.

    A NaN cell means the sensor never reported for this unit — the reading
    simply doesn't happen, leaving the engine's placeholder in that slot.
    """
    for pos, sensor in enumerate(keep):
        value = row[sensor]
        if not np.isnan(value):
            yield (
                f"station_{pos // SENSORS_PER_STATION + 1:02d}",
                f"s{sensor:03d}",
                float(value),
            )
