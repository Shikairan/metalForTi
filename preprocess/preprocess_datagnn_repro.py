#!/usr/bin/env python3
"""复现 dataOri2 <-> datagnn 正向预处理与逆变换。详见 PREPROCESS_datagnn.md。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple

import numpy as np

ELEMENT_COLS = ["Al", "Zr", "Sn", "Mo", "Cr", "Nb", "Si", "V", "Ta", "Fe"]
TESTENV_COLS = ["tem", "fcr"]
TARGET_COLS = ["YS", "FS"]
COLDWAY_COLS = [
    "T1", "t1", "T2", "t2", "T3", "t3",
    "C1_1", "C1_2", "C1_3", "C2_1", "C2_2", "C2_3", "C3_1", "C3_2", "C3_3",
]
EPS = 1e-8
T_DIV = 800.0

DEFAULT_TESTENV_MEAN = np.array([194.7433774834437, 1342.7152317880796], dtype=np.float64)
DEFAULT_TESTENV_STD = np.array([214.62536752046483, 1547.0319516107018], dtype=np.float64)
DEFAULT_YS_MEAN = 965.7821034430465
DEFAULT_FS_MEAN = 28.120464644701983


def normalize_coldway_tx_cx(raw: np.ndarray) -> np.ndarray:
    x = np.asarray(raw, dtype=np.float32)
    single = x.ndim == 1
    if single:
        x = x.reshape(1, -1)
    scaled = x.copy()
    t_idx = {0, 2, 4}
    for j in range(6):
        col = x[:, j]
        mask = col != -1.0
        if not np.any(mask):
            continue
        if j in t_idx:
            scaled[mask, j] = np.log(scaled[mask, j] / np.float32(T_DIV))
        else:
            scaled[mask, j] = np.log(scaled[mask, j])
    scaled[:, 6:15] = x[:, 6:15]
    return scaled[0] if single else scaled


def coldway_row_to_seq_3x6(raw: np.ndarray, scaled: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    scaled = np.asarray(scaled, dtype=np.float32)
    C = raw[6:15].reshape(3, 3)
    out = np.zeros((3, 3, 2), dtype=np.float32)
    tt_raw = [(raw[0], raw[1]), (raw[2], raw[3]), (raw[4], raw[5])]
    tt_scaled = [(scaled[0], scaled[1]), (scaled[2], scaled[3]), (scaled[4], scaled[5])]
    for i in range(3):
        T_r, t_r = tt_raw[i]
        invalid = (T_r == -1.0) or (t_r == -1.0)
        Ts, ts = tt_scaled[i]
        for j in range(3):
            if float(C[i, j]) != 1.0:
                continue
            if invalid:
                out[i, j, :] = 0.0
            else:
                out[i, j, 0] = np.float32(Ts)
                out[i, j, 1] = np.float32(ts)
    return out.reshape(3, 6).reshape(-1)


def coldway_seq_to_raw_15(flat18: np.ndarray) -> np.ndarray:
    cell = np.asarray(flat18, dtype=np.float32).reshape(3, 6).reshape(3, 3, 2)
    raw = np.full(15, -1.0, dtype=np.float32)
    for i in range(3):
        t_idx = i * 2
        T_i, t_i = -1.0, -1.0
        C_row = np.zeros(3, dtype=np.float32)
        for j in range(3):
            Ts, ts = float(cell[i, j, 0]), float(cell[i, j, 1])
            if Ts != 0.0 or ts != 0.0:
                C_row[j] = 1.0
                if T_i == -1.0:
                    T_i = T_DIV * np.exp(Ts)
                    t_i = np.exp(ts)
        raw[t_idx] = np.float32(T_i)
        raw[t_idx + 1] = np.float32(t_i)
        raw[6 + i * 3 : 6 + i * 3 + 3] = C_row
    return raw


def read_dataori(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        el, te, cw, tg = [], [], [], []
        for r in reader:
            if not r:
                continue
            el.append([float(r[c]) for c in ELEMENT_COLS])
            te.append([float(r[c]) for c in TESTENV_COLS])
            cw.append([float(r[c]) for c in COLDWAY_COLS])
            tg.append([float(r[c]) for c in TARGET_COLS])
    return (
        np.asarray(el, dtype=np.float32),
        np.asarray(te, dtype=np.float32),
        np.asarray(cw, dtype=np.float32),
        np.asarray(tg, dtype=np.float32),
    )


def forward_preprocess(
    element: np.ndarray,
    testenv: np.ndarray,
    coldway_raw: np.ndarray,
    targets: np.ndarray,
    *,
    te_mean: np.ndarray = DEFAULT_TESTENV_MEAN,
    te_std: np.ndarray = DEFAULT_TESTENV_STD,
    normalize_targets: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = element.shape[0]
    element_out = element.astype(np.float32, copy=False)
    std_safe = np.where(te_std == 0, 1.0, te_std).astype(np.float64)
    testenv_out = ((testenv.astype(np.float64) - te_mean) / std_safe).astype(np.float32)
    coldway_out = np.zeros((n, 18), dtype=np.float32)
    scaled_all = normalize_coldway_tx_cx(coldway_raw)
    for i in range(n):
        coldway_out[i] = coldway_row_to_seq_3x6(coldway_raw[i], scaled_all[i])
    targets_out = targets.astype(np.float32, copy=True)
    if normalize_targets:
        targets_out[:, 0] /= np.float32(targets[:, 0].mean() + EPS)
        targets_out[:, 1] /= np.float32(targets[:, 1].mean() + EPS)
    return element_out, testenv_out, coldway_out, targets_out


def inverse_preprocess(
    element: np.ndarray,
    testenv_z: np.ndarray,
    coldway_flat: np.ndarray,
    targets_norm: np.ndarray,
    *,
    te_mean: np.ndarray = DEFAULT_TESTENV_MEAN,
    te_std: np.ndarray = DEFAULT_TESTENV_STD,
    ys_mean: float = DEFAULT_YS_MEAN,
    fs_mean: float = DEFAULT_FS_MEAN,
    denormalize_targets: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    std_safe = np.where(te_std == 0, 1.0, te_std).astype(np.float64)
    testenv_raw = (testenv_z.astype(np.float64) * std_safe + te_mean).astype(np.float32)
    n = coldway_flat.shape[0]
    coldway_raw = np.stack([coldway_seq_to_raw_15(coldway_flat[i]) for i in range(n)], axis=0)
    targets_raw = targets_norm.astype(np.float32, copy=True)
    if denormalize_targets:
        targets_raw[:, 0] *= np.float32(ys_mean + EPS)
        targets_raw[:, 1] *= np.float32(fs_mean + EPS)
    return element.astype(np.float32, copy=False), testenv_raw, coldway_raw, targets_raw


def write_datagnn(
    path: Path,
    element: np.ndarray,
    testenv: np.ndarray,
    coldway: np.ndarray,
    targets: np.ndarray,
) -> None:
    header = (
        [f"element_{i}" for i in range(10)]
        + [f"testenv_{i}" for i in range(2)]
        + [f"coldway_{i}" for i in range(18)]
        + ["YS", "FS"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(element.shape[0]):
            w.writerow(
                element[i].tolist()
                + testenv[i].tolist()
                + coldway[i].tolist()
                + targets[i].tolist()
            )


def read_datagnn(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        el, te, cw, tg = [], [], [], []
        for r in reader:
            el.append([float(r[f"element_{i}"]) for i in range(10)])
            te.append([float(r[f"testenv_{i}"]) for i in range(2)])
            cw.append([float(r[f"coldway_{i}"]) for i in range(18)])
            tg.append([float(r["YS"]), float(r["FS"])])
    return (
        np.asarray(el, dtype=np.float32),
        np.asarray(te, dtype=np.float32),
        np.asarray(cw, dtype=np.float32),
        np.asarray(tg, dtype=np.float32),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="dataOri2 <-> datagnn 预处理复现")
    p.add_argument("mode", choices=["forward", "inverse", "verify"])
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--reference", type=Path, default=None, help="verify 时与 datagnn.csv 对比")
    args = p.parse_args()

    if args.mode == "forward":
        el, te, cw, tg = read_dataori(args.input)
        out = forward_preprocess(el, te, cw, tg, normalize_targets=True)
        out_path = args.output or args.input.parent / "datagnn_repro.csv"
        write_datagnn(out_path, *out)
        print(f"[OK] forward -> {out_path} rows={el.shape[0]}")

    elif args.mode == "inverse":
        el, te_z, cw_f, tg_n = read_datagnn(args.input)
        el, te, cw, tg = inverse_preprocess(el, te_z, cw_f, tg_n)
        out_path = args.output or args.input.parent / "dataOri_restored.csv"
        header = TESTENV_COLS + ELEMENT_COLS + COLDWAY_COLS + TARGET_COLS
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(el.shape[0]):
                w.writerow(te[i].tolist() + el[i].tolist() + cw[i].tolist() + tg[i].tolist())
        print(f"[OK] inverse -> {out_path} rows={el.shape[0]}")

    else:
        el, te, cw, tg = read_dataori(args.input)
        pred = forward_preprocess(el, te, cw, tg, normalize_targets=True)
        ref_path = args.reference or Path(__file__).parent / "datagnn.csv"
        ref = read_datagnn(ref_path)
        names = ["element", "testenv", "coldway", "targets"]
        ok = True
        for a, b, name in zip(pred, ref, names):
            m = np.allclose(a, b, rtol=1e-5, atol=1e-5, equal_nan=True)
            print(f"{name}: match={m}")
            ok = ok and m
        print("ALL OK" if ok else "MISMATCH")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
