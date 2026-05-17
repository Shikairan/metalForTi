#!/usr/bin/env python3
"""
One command to refresh gnndataPT artifacts for r-gnn or r-gat:

1) build_datagnn.py  -> datacsv/datagnn.csv (+ testenv_stats.csv next to it)
2) rgcn_dataloader.py -> material_graph.pt, ys.pt, fs.pt
3) Export train_mask.pt / val_mask.pt from the saved HeteroData (same split as in graph)

Default output dir: --pt-bundle rgnn -> gnndataPT/r-gnnPT; rgat -> gnndataPT/r-gatPT.
Override with --rgnnpt-dir.

After a successful run, removes that bundle's mask-loop state + history under gnn/<r-gnn|r-gat>/runs/
so the next loop_train_swap_* starts from round 1 (unless --keep-loop-state).
Does not modify loop script source files.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


def _base_dir() -> Path:
    return Path(__file__).resolve().parent


def _pt_dir_for_bundle(base: Path, bundle: str) -> Path:
    if bundle == "rgnn":
        return base / "gnndataPT" / "r-gnnPT"
    if bundle == "rgat":
        return base / "gnndataPT" / "r-gatPT"
    raise ValueError(f"unknown pt-bundle: {bundle!r}")


def _clear_mask_loop_artifacts(all_test_part: Path, bundle: str) -> None:
    """Remove loop state JSON and swap history CSV so the next loop starts at round 1."""
    if bundle == "rgnn":
        runs = all_test_part / "gnn" / "r-gnn" / "runs"
        names = ("rgcn_mask_loop_state.json", "rgcn_mask_swap_history.csv")
    elif bundle == "rgat":
        runs = all_test_part / "gnn" / "r-gat" / "runs"
        names = ("gat_mask_loop_state.json", "gat_mask_swap_history.csv")
    else:
        raise ValueError(f"unknown pt-bundle: {bundle!r}")
    for name in names:
        p = runs / name
        if p.is_file():
            p.unlink()
            print(f"[OK] removed loop artifact {p}")
        else:
            print(f"[SKIP] no file {p}")


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def _load_graph(graph_path: Path):
    try:
        return torch.load(graph_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(graph_path, map_location="cpu")


def _export_masks_from_data(data, out_train: Path, out_val: Path) -> None:
    tr = data["sample"].train_mask.detach().cpu().bool().reshape(-1)
    va = data["sample"].val_mask.detach().cpu().bool().reshape(-1)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tr, out_train)
    torch.save(va, out_val)
    print(f"[OK] exported {out_train} sum={int(tr.sum())}")
    print(f"[OK] exported {out_val} sum={int(va.sum())}")


def _print_new_graph_summary(data) -> None:
    n = int(data["sample"].x.shape[0])
    n_comp = int(data[("sample", "comp_sim", "sample")].edge_index.shape[1])
    n_env = int(data[("sample", "env_sim", "sample")].edge_index.shape[1])
    n_heat = int(data[("sample", "heat_sim", "sample")].edge_index.shape[1])
    print(
        f"[SUMMARY] num_nodes={n} "
        f"comp_sim_edges={n_comp} env_sim_edges={n_env} heat_sim_edges={n_heat}"
    )


def parse_args() -> argparse.Namespace:
    b = _base_dir()
    default_datagnn = b / "datacsv" / "datagnn.csv"
    default_dataori = b.parent / "dataOri.csv"

    p = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    p.add_argument(
        "--dataori",
        type=Path,
        default=default_dataori,
        help="Input for build_datagnn (default: symtest/dataOri.csv).",
    )
    p.add_argument(
        "--datagnn-csv",
        type=Path,
        default=default_datagnn,
        help="Output datagnn.csv for build_datagnn (default: gnnDir/datacsv/datagnn.csv).",
    )
    p.add_argument(
        "--pt-bundle",
        choices=["rgnn", "rgat"],
        default="rgnn",
        help="Write under gnndataPT/: rgnn -> r-gnnPT, rgat -> r-gatPT (ignored if --rgnnpt-dir is set).",
    )
    p.add_argument(
        "--rgnnpt-dir",
        type=Path,
        default=None,
        help="Explicit directory for the 5 .pt files (overrides --pt-bundle).",
    )
    p.add_argument(
        "--skip-build-datagnn",
        action="store_true",
        help="Only run rgcn_dataloader + mask export (reuse existing datagnn.csv).",
    )
    p.add_argument(
        "--sanity",
        action="store_true",
        help="Pass --sanity to rgcn_dataloader after build.",
    )
    p.add_argument("--element-thr", type=float, default=None)
    p.add_argument("--testenv-thr", type=float, default=None)
    p.add_argument("--coldway-thr", type=float, default=None)
    p.add_argument("--train-ratio", type=float, default=None)
    p.add_argument("--split-seed", type=int, default=None)
    p.add_argument(
        "--train-mask-path",
        type=Path,
        default=None,
        help="If set with --val-mask-path, forwarded to rgcn_dataloader (otherwise random split).",
    )
    p.add_argument("--val-mask-path", type=Path, default=None)
    p.add_argument(
        "--keep-loop-state",
        action="store_true",
        help="Do not delete gnn/*/runs/*_mask_loop_state.json or *_mask_swap_history.csv.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    b = _base_dir()
    py = sys.executable
    build_script = b / "build_datagnn.py"
    dataloader_script = b / "rgcn_dataloader.py"

    if not args.skip_build_datagnn:
        _run(
            [
                py,
                str(build_script),
                "--input",
                str(args.dataori.resolve()),
                "--output",
                str(args.datagnn_csv.resolve()),
            ]
        )

    rdir = (
        args.rgnnpt_dir.resolve()
        if args.rgnnpt_dir is not None
        else _pt_dir_for_bundle(b, args.pt_bundle).resolve()
    )
    print(f"[INIT] pt_bundle={args.pt_bundle} out_pt_dir={rdir}")
    rdir.mkdir(parents=True, exist_ok=True)
    graph_pt = rdir / "material_graph.pt"
    ys_pt = rdir / "ys.pt"
    fs_pt = rdir / "fs.pt"
    train_pt = rdir / "train_mask.pt"
    val_pt = rdir / "val_mask.pt"

    dl_cmd: list[str] = [
        py,
        str(dataloader_script),
        "--csv",
        str(args.datagnn_csv.resolve()),
        "--out-graph",
        str(graph_pt),
        "--out-ys",
        str(ys_pt),
        "--out-fs",
        str(fs_pt),
    ]
    if args.element_thr is not None:
        dl_cmd += ["--element-thr", str(args.element_thr)]
    if args.testenv_thr is not None:
        dl_cmd += ["--testenv-thr", str(args.testenv_thr)]
    if args.coldway_thr is not None:
        dl_cmd += ["--coldway-thr", str(args.coldway_thr)]
    if args.train_ratio is not None:
        dl_cmd += ["--train-ratio", str(args.train_ratio)]
    if args.split_seed is not None:
        dl_cmd += ["--split-seed", str(args.split_seed)]
    if args.train_mask_path is not None:
        dl_cmd += ["--train-mask-path", str(args.train_mask_path.resolve())]
    if args.val_mask_path is not None:
        dl_cmd += ["--val-mask-path", str(args.val_mask_path.resolve())]
    if args.sanity:
        dl_cmd.append("--sanity")

    _run(dl_cmd)
    data = _load_graph(graph_pt)
    _export_masks_from_data(data, train_pt, val_pt)
    _print_new_graph_summary(data)

    print(
        "[OK] PT bundle:",
        graph_pt.name,
        ys_pt.name,
        fs_pt.name,
        train_pt.name,
        val_pt.name,
    )

    if not args.keep_loop_state:
        print(f"[INIT] clearing mask-loop state/history for pt_bundle={args.pt_bundle}")
        _clear_mask_loop_artifacts(b, args.pt_bundle)


if __name__ == "__main__":
    main()
