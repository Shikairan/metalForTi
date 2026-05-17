from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def _pairwise_cosine_sim_zero_equal_one(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Pairwise cosine similarity with custom rule:
    if both vectors are zero-norm (<eps), similarity is forced to 1.
    """
    x = mat.astype(np.float32, copy=False)
    norm_mat = _l2_normalize(x, eps=eps)
    sim = norm_mat @ norm_mat.T
    norms = np.linalg.norm(x, axis=1)
    is_zero = norms < float(eps)
    if bool(is_zero.any()):
        zz = np.outer(is_zero, is_zero)
        sim[zz] = 1.0
    return sim


def _similarity_mask_upper_from_sim(sim: np.ndarray, thr: float) -> np.ndarray:
    n = sim.shape[0]
    iu = np.triu_indices(n, k=1)
    m = np.zeros((n, n), dtype=bool)
    m[iu] = sim[iu] > float(thr)
    return m


def _upper_mask_to_bidirectional_edge_index(mask_upper: np.ndarray) -> np.ndarray:
    src, dst = np.where(mask_upper)
    if src.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    e_fwd = np.vstack([src, dst])
    e_bwd = np.vstack([dst, src])
    return np.concatenate([e_fwd, e_bwd], axis=1)


def _build_relation_edges(
    element: np.ndarray,
    testenv: np.ndarray,
    coldway_flat: np.ndarray,
    *,
    element_thr: float,
    testenv_thr: float,
    coldway_thr: float,
) -> Dict[str, torch.Tensor]:
    n = element.shape[0]
    if not (testenv.shape[0] == n and coldway_flat.shape[0] == n):
        raise ValueError("Feature rows mismatch among element/testenv/coldway")

    # comp_sim: cosine on element
    comp_sim = _pairwise_cosine_sim_zero_equal_one(element)
    comp_mask = _similarity_mask_upper_from_sim(comp_sim, float(element_thr))
    comp_edge = _upper_mask_to_bidirectional_edge_index(comp_mask)

    # env_sim: cosine on testenv
    env_sim = _pairwise_cosine_sim_zero_equal_one(testenv)
    env_mask = _similarity_mask_upper_from_sim(env_sim, float(testenv_thr))
    env_edge = _upper_mask_to_bidirectional_edge_index(env_mask)

    # heat_sim: coldway 18 -> (3,6), cosine row-wise mean
    c = coldway_flat.astype(np.float32).reshape(n, 3, 6)
    s0 = _pairwise_cosine_sim_zero_equal_one(c[:, 0, :])
    s1 = _pairwise_cosine_sim_zero_equal_one(c[:, 1, :])
    s2 = _pairwise_cosine_sim_zero_equal_one(c[:, 2, :])
    sim_mean = (s0 + s1 + s2) / 3.0
    heat_mask = _similarity_mask_upper_from_sim(sim_mean, float(coldway_thr))
    heat_edge = _upper_mask_to_bidirectional_edge_index(heat_mask)

    return {
        "comp_sim": torch.tensor(comp_edge, dtype=torch.long),
        "env_sim": torch.tensor(env_edge, dtype=torch.long),
        "heat_sim": torch.tensor(heat_edge, dtype=torch.long),
    }


def _load_mask_file(mask_path: Path, n: int) -> torch.Tensor:
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    suffix = mask_path.suffix.lower()

    if suffix == ".pt":
        v = torch.load(mask_path, map_location="cpu")
        if isinstance(v, dict):
            if "mask" in v:
                v = v["mask"]
            else:
                raise ValueError(f"PT mask dict should contain key 'mask': {mask_path}")
        mask = torch.as_tensor(v)
    elif suffix == ".npy":
        arr = np.load(mask_path)
        mask = torch.from_numpy(arr)
    elif suffix == ".csv":
        df = pd.read_csv(mask_path)
        if "mask" in df.columns:
            arr = df["mask"].to_numpy()
        elif "value" in df.columns:
            arr = df["value"].to_numpy()
        else:
            arr = df.iloc[:, 0].to_numpy()
        mask = torch.from_numpy(arr)
    else:
        raise ValueError(f"Unsupported mask file format: {mask_path}")

    mask = mask.reshape(-1).bool()
    if mask.numel() != n:
        raise ValueError(f"Mask size mismatch: expect {n}, got {mask.numel()} from {mask_path}")
    return mask


def _build_masks(
    n: int,
    *,
    train_mask_path: Path | None,
    val_mask_path: Path | None,
    train_ratio: float,
    split_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if train_mask_path is not None and val_mask_path is not None:
        train_mask = _load_mask_file(train_mask_path, n)
        val_mask = _load_mask_file(val_mask_path, n)
    elif train_mask_path is None and val_mask_path is None:
        if not (0.0 < train_ratio < 1.0):
            raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")
        rng = np.random.default_rng(int(split_seed))
        perm = rng.permutation(n)
        n_train = int(round(n * float(train_ratio)))
        n_train = max(1, min(n - 1, n_train))
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[torch.from_numpy(train_idx)] = True
        val_mask[torch.from_numpy(val_idx)] = True
    else:
        raise ValueError("Please provide both train_mask_path and val_mask_path, or provide neither.")

    overlap = train_mask & val_mask
    if bool(overlap.any()):
        raise ValueError("train_mask and val_mask overlap.")
    union = train_mask | val_mask
    if not bool(union.all()):
        raise ValueError("train_mask and val_mask do not cover all nodes.")
    if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0:
        raise ValueError("train_mask and val_mask must both be non-empty.")
    return train_mask, val_mask


def _load_datagnn_components(csv_path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    elem_cols = [f"element_{i}" for i in range(10)]
    tenv_cols = [f"testenv_{i}" for i in range(2)]
    cold_cols = [f"coldway_{i}" for i in range(18)]
    need = elem_cols + tenv_cols + cold_cols + ["YS", "FS"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in {csv_path}: {miss}")

    element = df[elem_cols].to_numpy(dtype=np.float32)
    testenv = df[tenv_cols].to_numpy(dtype=np.float32)
    coldway = df[cold_cols].to_numpy(dtype=np.float32)

    x = torch.tensor(np.concatenate([element, testenv, coldway], axis=1), dtype=torch.float32)
    ys = torch.tensor(df["YS"].to_numpy(dtype=np.float32), dtype=torch.float32)
    fs = torch.tensor(df["FS"].to_numpy(dtype=np.float32), dtype=torch.float32)

    rel_edges = _build_relation_edges(
        element,
        testenv,
        coldway,
        element_thr=0.6,
        testenv_thr=0.6,
        coldway_thr=0.75,
    )
    return x, ys, fs, rel_edges


def build_material_heterodata(
    csv_path: Path,
    *,
    element_thr: float,
    testenv_thr: float,
    coldway_thr: float,
    train_mask_path: Path | None,
    val_mask_path: Path | None,
    train_ratio: float,
    split_seed: int,
):
    try:
        from torch_geometric.data import HeteroData  # type: ignore
    except Exception as e:
        raise RuntimeError("torch_geometric is required to build HeteroData material_graph.pt") from e

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    elem_cols = [f"element_{i}" for i in range(10)]
    tenv_cols = [f"testenv_{i}" for i in range(2)]
    cold_cols = [f"coldway_{i}" for i in range(18)]
    need = elem_cols + tenv_cols + cold_cols + ["YS", "FS"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in {csv_path}: {miss}")

    element = df[elem_cols].to_numpy(dtype=np.float32)
    testenv = df[tenv_cols].to_numpy(dtype=np.float32)
    coldway = df[cold_cols].to_numpy(dtype=np.float32)
    x = torch.tensor(np.concatenate([element, testenv, coldway], axis=1), dtype=torch.float32)
    ys = torch.tensor(df["YS"].to_numpy(dtype=np.float32), dtype=torch.float32)
    fs = torch.tensor(df["FS"].to_numpy(dtype=np.float32), dtype=torch.float32)

    rel_edges = _build_relation_edges(
        element,
        testenv,
        coldway,
        element_thr=element_thr,
        testenv_thr=testenv_thr,
        coldway_thr=coldway_thr,
    )
    n = x.shape[0]
    train_mask, val_mask = _build_masks(
        n,
        train_mask_path=train_mask_path,
        val_mask_path=val_mask_path,
        train_ratio=train_ratio,
        split_seed=split_seed,
    )

    data = HeteroData()
    data["sample"].x = x
    data["sample"].train_mask = train_mask
    data["sample"].val_mask = val_mask
    data[("sample", "comp_sim", "sample")].edge_index = rel_edges["comp_sim"]
    data[("sample", "env_sim", "sample")].edge_index = rel_edges["env_sim"]
    data[("sample", "heat_sim", "sample")].edge_index = rel_edges["heat_sim"]
    data["sample"].num_nodes = int(n)

    return data, ys, fs


def _edge_sanity(edge_index: torch.Tensor) -> Tuple[int, int]:
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    self_loops = int(np.sum(src == dst))
    pairs = set(zip(src.tolist(), dst.tolist()))
    miss = 0
    for a, b in pairs:
        if (b, a) not in pairs:
            miss += 1
            if miss >= 10:
                break
    return self_loops, miss


def print_sanity(data, ys: torch.Tensor, fs: torch.Tensor) -> None:
    x = data["sample"].x
    train_mask = data["sample"].train_mask
    val_mask = data["sample"].val_mask
    c = data[("sample", "comp_sim", "sample")].edge_index
    e = data[("sample", "env_sim", "sample")].edge_index
    h = data[("sample", "heat_sim", "sample")].edge_index

    print(f"x.shape={tuple(x.shape)} ys.shape={tuple(ys.shape)} fs.shape={tuple(fs.shape)}")
    print(f"train_mask.sum={int(train_mask.sum())} val_mask.sum={int(val_mask.sum())}")
    print(f"comp_sim edges={c.shape[1]} env_sim edges={e.shape[1]} heat_sim edges={h.shape[1]}")

    for name, edge in [("comp_sim", c), ("env_sim", e), ("heat_sim", h)]:
        self_loops, miss = _edge_sanity(edge)
        print(f"{name}: self_loops={self_loops}, symmetry_missing_pairs={miss}")


def print_train_usage(out_graph: Path, out_ys: Path, out_fs: Path) -> None:
    usage = f"""
Minimal training read example:
  import torch
  graph = torch.load(r"{out_graph}", map_location="cpu")
  ys = torch.load(r"{out_ys}", map_location="cpu")
  fs = torch.load(r"{out_fs}", map_location="cpu")

  x = graph["sample"].x
  train_mask = graph["sample"].train_mask
  val_mask = graph["sample"].val_mask

  # For YS:
  x_train, ys_train = x[train_mask], ys[train_mask]
  x_val, ys_val = x[val_mask], ys[val_mask]

  # For FS:
  fs_train = fs[train_mask]
  fs_val = fs[val_mask]
"""
    print(usage.strip())


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    default_csv = base / "datacsv" / "datagnn.csv"
    default_out_dir = base / "gnndataPT" / "r-gnnPT"
    p = argparse.ArgumentParser(description="Export HeteroData graph and labels from datagnn.csv")
    p.add_argument("--csv", type=Path, default=default_csv)
    p.add_argument("--out-graph", type=Path, default=default_out_dir / "material_graph.pt")
    p.add_argument("--out-ys", type=Path, default=default_out_dir / "ys.pt")
    p.add_argument("--out-fs", type=Path, default=default_out_dir / "fs.pt")
    p.add_argument("--element-thr", type=float, default=0.8)
    p.add_argument("--testenv-thr", type=float, default=0.8)
    p.add_argument("--coldway-thr", type=float, default=0.8)
    p.add_argument("--train-mask-path", type=Path, default=None)
    p.add_argument("--val-mask-path", type=Path, default=None)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.80)
    p.add_argument("--sanity", action="store_true")
    p.add_argument("--show-train-usage", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data, ys, fs = build_material_heterodata(
        args.csv,
        element_thr=float(args.element_thr),
        testenv_thr=float(args.testenv_thr),
        coldway_thr=float(args.coldway_thr),
        train_mask_path=args.train_mask_path,
        val_mask_path=args.val_mask_path,
        train_ratio=float(args.train_ratio),
        split_seed=int(args.split_seed),
    )

    args.out_graph.parent.mkdir(parents=True, exist_ok=True)
    args.out_ys.parent.mkdir(parents=True, exist_ok=True)
    args.out_fs.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.out_graph)
    torch.save(ys, args.out_ys)
    torch.save(fs, args.out_fs)

    print(f"Saved graph: {args.out_graph}")
    print(f"Saved labels: {args.out_ys}, {args.out_fs}")
    if args.sanity:
        print_sanity(data, ys, fs)
    if args.show_train_usage:
        print_train_usage(args.out_graph, args.out_ys, args.out_fs)


if __name__ == "__main__":
    main()

