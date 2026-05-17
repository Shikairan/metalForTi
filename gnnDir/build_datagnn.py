from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

import numpy as np

# Reuse preprocessing/utilities from pt_dataset.py (same repo)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pt_dataset import (
    FIELD_PREPROCESS_MODES,
    _build_modes_from_field_dict,
    _coldway_row_to_seq_3x6,
    _normalize_coldway_tx_cx,
    _preprocess_columns_with_modes,
)


ELEMENT_COLS: List[str] = ["Al", "Zr", "Sn", "Mo", "Cr", "Nb", "Si", "V", "Ta", "Fe"]
TESTENV_COLS: List[str] = ["tem", "fcr"]
TARGET_COLS: List[str] = ["YS", "FS"]
COLDWAY_COLS: List[str] = [
    "T1",
    "t1",
    "T2",
    "t2",
    "T3",
    "t3",
    "C1_1",
    "C1_2",
    "C1_3",
    "C2_1",
    "C2_2",
    "C2_3",
    "C3_1",
    "C3_2",
    "C3_3",
]


def _read_dataori_numeric(input_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with input_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_csv}")
        missing = [c for c in (ELEMENT_COLS + TESTENV_COLS + COLDWAY_COLS + TARGET_COLS) if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing columns in {input_csv}: {missing}")

        element_rows: List[List[float]] = []
        testenv_rows: List[List[float]] = []
        coldway_rows: List[List[float]] = []
        target_rows: List[List[float]] = []

        for r in reader:
            if not r:
                continue
            # element
            element_rows.append([float(r[c]) for c in ELEMENT_COLS])
            testenv_rows.append([float(r[c]) for c in TESTENV_COLS])
            coldway_rows.append([float(r[c]) for c in COLDWAY_COLS])
            target_rows.append([float(r[c]) for c in TARGET_COLS])

    if not element_rows:
        raise ValueError(f"No data rows found in {input_csv}")
    element = np.asarray(element_rows, dtype=np.float32)
    testenv = np.asarray(testenv_rows, dtype=np.float32)
    coldway_raw = np.asarray(coldway_rows, dtype=np.float32)
    targets = np.asarray(target_rows, dtype=np.float32)
    if element.shape[1] != len(ELEMENT_COLS):
        raise ValueError(f"element shape mismatch: got {element.shape}")
    if testenv.shape[1] != len(TESTENV_COLS):
        raise ValueError(f"testenv shape mismatch: got {testenv.shape}")
    if coldway_raw.shape[1] != len(COLDWAY_COLS):
        raise ValueError(f"coldway_raw shape mismatch: got {coldway_raw.shape}")
    if targets.shape[1] != len(TARGET_COLS):
        raise ValueError(f"targets shape mismatch: got {targets.shape}")
    return element, testenv, coldway_raw, targets


def _preprocess_element(element_data: np.ndarray) -> np.ndarray:
    # Align with pt_dataset defaults: element_preprocess="mean", but FIELD_PREPROCESS_MODES
    # currently sets element cols to "nofix" so this is effectively a float32 cast.
    modes = _build_modes_from_field_dict(
        column_names=ELEMENT_COLS,
        default_mode="mean",
        preprocess_by_field=FIELD_PREPROCESS_MODES,
    )
    out = _preprocess_columns_with_modes(
        element_data,
        default_mode="mean",
        per_col_modes=modes,
        eps=1e-8,
        log_t=1.0,
    )
    return out.astype(np.float32)


def _preprocess_coldway_to_seq_flat(coldway_raw: np.ndarray) -> np.ndarray:
    # coldway_raw: (N, 15) in pt_dataset fixed column order
    coldway_scaled = _normalize_coldway_tx_cx(coldway_raw, eps=1e-8)
    n = int(coldway_raw.shape[0])
    seq_flat = np.zeros((n, 18), dtype=np.float32)
    for i in range(n):
        seq = _coldway_row_to_seq_3x6(coldway_raw[i], coldway_scaled[i])  # (3,6)
        seq_flat[i, :] = np.asarray(seq, dtype=np.float32).reshape(-1)
    return seq_flat


def build_datagnn_csv(
    *,
    input_csv: Path,
    output_csv: Path,
) -> None:
    element, testenv, coldway_raw, targets = _read_dataori_numeric(input_csv)
    element_pre = _preprocess_element(element)
    coldway_flat = _preprocess_coldway_to_seq_flat(coldway_raw)

    # Standardize testenv per column: (x - mean) / std, keep shape 1x2.
    # Stats are written alongside datagnn.csv so predictions can be inverse-transformed.
    testenv = testenv.astype(np.float32, copy=False)
    means = testenv.mean(axis=0, keepdims=True)
    stds = testenv.std(axis=0, ddof=0, keepdims=True)
    # Avoid division by zero: if std==0, fall back to 1 (all values identical).
    stds_safe = np.where(stds == 0.0, 1.0, stds).astype(np.float32)
    testenv_out = (testenv - means) / stds_safe

    if (
        element_pre.shape[0] != testenv_out.shape[0]
        or element_pre.shape[0] != coldway_flat.shape[0]
        or element_pre.shape[0] != targets.shape[0]
    ):
        raise ValueError("Row count mismatch after preprocessing")
    n = int(element_pre.shape[0])

    # Write per-sample single row: element(10) + testenv(2) + coldway(18) + targets(2) = 32 columns
    header = (
        [f"element_{i}" for i in range(10)]
        + [f"testenv_{i}" for i in range(2)]
        + [f"coldway_{i}" for i in range(18)]
        + ["YS", "FS"]
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(n):
            row = (
                element_pre[i, :].tolist()
                + testenv_out[i, :].tolist()
                + coldway_flat[i, :].tolist()
                + targets[i, :].tolist()
            )
            w.writerow(row)

    print(f"[OK] wrote {output_csv} rows={n} cols={len(header)}")

    # Write testenv standardization stats for inverse transform: x = z * std + mean
    stats_path = output_csv.with_name("testenv_stats.csv")
    with stats_path.open("w", newline="", encoding="utf-8") as sf:
        sw = csv.writer(sf)
        sw.writerow(["col", "mean", "std"])
        for j, name in enumerate(TESTENV_COLS):
            sw.writerow([name, float(means[0, j]), float(stds_safe[0, j])])
    print(f"[OK] wrote testenv stats to {stats_path}")


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    default_out_dir = base / "datacsv"
    p = argparse.ArgumentParser(
        description=(
            "Convert dataOri.csv to datagnn.csv with element(1x10), testenv(1x2), coldway(3x6->18) flattened.\n"
            "Reuses pt_dataset preprocessing for element and coldway; testenv is column-wise standardized\n"
            "to zero mean and unit variance, with stats written to testenv_stats.csv for inverse transform."
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        default=base.parent / "dataOri.csv",
        help="Input dataOri.csv path (default: ../dataOri.csv).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=default_out_dir / "datagnn.csv",
        help="Output datagnn.csv path (default: ./datacsv/datagnn.csv).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"--input not found: {args.input}")
    build_datagnn_csv(input_csv=args.input, output_csv=args.output)


if __name__ == "__main__":
    main()

