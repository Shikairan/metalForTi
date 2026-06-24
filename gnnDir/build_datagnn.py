from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess.preprocess_datagnn_repro import (
    coldway_row_to_seq_3x6,
    normalize_coldway_tx_cx,
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
    return element, testenv, coldway_raw, targets


def _preprocess_coldway_to_seq_flat(coldway_raw: np.ndarray) -> np.ndarray:
    coldway_scaled = normalize_coldway_tx_cx(coldway_raw)
    n = int(coldway_raw.shape[0])
    seq_flat = np.zeros((n, 18), dtype=np.float32)
    for i in range(n):
        seq = coldway_row_to_seq_3x6(coldway_raw[i], coldway_scaled[i])
        seq_flat[i, :] = np.asarray(seq, dtype=np.float32).reshape(-1)
    return seq_flat


def build_datagnn_csv(
    *,
    input_csv: Path,
    output_csv: Path,
) -> None:
    element, testenv, coldway_raw, targets = _read_dataori_numeric(input_csv)
    element_pre = element.astype(np.float32, copy=False)
    coldway_flat = _preprocess_coldway_to_seq_flat(coldway_raw)

    testenv = testenv.astype(np.float32, copy=False)
    means = testenv.mean(axis=0, keepdims=True)
    stds = testenv.std(axis=0, ddof=0, keepdims=True)
    stds_safe = np.where(stds == 0.0, 1.0, stds).astype(np.float32)
    testenv_out = (testenv - means) / stds_safe

    n = int(element_pre.shape[0])
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
            "Convert dataOri2-style CSV to datagnn.csv "
            "(element 10 + testenv z-score 2 + coldway 18 + YS/FS).\n"
            "Coldway 与 preprocess/preprocess_datagnn_repro.py 一致；"
            "testenv 按当前表列均值/标准差标准化，并写出 testenv_stats.csv。"
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        default=base.parent / "preprocess" / "dataOri2.csv",
        help="输入 CSV（默认 preprocess/dataOri2.csv）",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=default_out_dir / "datagnn.csv",
        help="输出 datagnn.csv（默认 gnnDir/datacsv/datagnn.csv）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"--input not found: {args.input}")
    build_datagnn_csv(input_csv=args.input, output_csv=args.output)


if __name__ == "__main__":
    main()
