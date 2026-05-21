#!/usr/bin/env python3
"""
run_inversion.py — 命令行入口：全图 GNN 输入特征梯度反推

联合反推 30 维：element(10) + testenv(2) + coldway(18)。
组分 A 模式：T_total=100 wt%，Ti = 100 - sum(element)。

函数:
  _resolve_device / _parse_args / _build_regularizers / main

默认 ground_truth ys/fs；优先 GPU。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import torch

from grd.feature_layout import (
    DEFAULT_TOTAL_WT,
    ELEMENT_NAMES,
    INPUT_DIM,
    bounds_from_train_x,
    bounds_with_physical_testenv,
    build_projector,
    compute_ti_balance,
)
from grd.gnn_inverter import (
    AnchorRegularizer,
    DirichletInitializer,
    GNNInverter,
    GNNInverterConfig,
    RandomNormalInitializer,
    SmoothnessRegularizer,
    SparsityRegularizer,
    TrainingMeanInitializer,
    ZeroInitializer,
)
from grd.io_utils import load_dual_rgat, load_graph_bundle, merge_hetero_edges
from grd.summary_report import build_summary_dict, write_summary_json, write_summary_txt

logger = logging.getLogger("grd.run_inversion")


def _resolve_device(requested: str) -> str:
    """
    解析运行设备：若请求 cuda 但不可用则回退 cpu 并打日志。

    参数:
        requested: 用户传入的 --device 字符串。

    返回:
        实际使用的设备名。
    """
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，已回退到 CPU")
        return "cpu"
    return requested


def _parse_args() -> argparse.Namespace:
    """
    构建命令行参数解析器并返回解析结果。

    默认路径相对于仓库根目录（grd 的上级）。
    """
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="GNN 全特征梯度反推 (grd)")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=root / "gnnDir" / "gnndataPT" / "r-gatPT",
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        default=root / "gnnDir" / "gnn" / "r-gatDouble" / "runs" / "best_ysfs_gat.pt",
    )
    p.add_argument(
        "--rgat-dir",
        type=Path,
        default=root / "gnnDir" / "gnn" / "r-gatDouble",
    )
    p.add_argument(
        "--testenv-stats",
        type=Path,
        default=root / "gnnDir" / "datacsv" / "testenv_stats.csv",
    )
    p.add_argument("--out-dir", type=Path, default=root / "grd" / "outputs")
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--total-wt", type=float, default=DEFAULT_TOTAL_WT, help="Ti 余量标尺 wt%")
    p.add_argument("--max-iters", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lambda-smooth", type=float, default=0.08)
    p.add_argument("--lambda-anchor", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--recon-tol", type=float, default=1e-5)
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--tem-lower", type=float, default=None)
    p.add_argument("--tem-upper", type=float, default=None)
    p.add_argument("--fcr-lower", type=float, default=None)
    p.add_argument("--fcr-upper", type=float, default=None)
    p.add_argument(
        "--target-mode",
        choices=["ground_truth", "model_forward"],
        default="ground_truth",
    )
    p.add_argument(
        "--node-mask",
        choices=["all", "val", "train"],
        default="val",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inits", type=str, default="")
    p.add_argument(
        "--force-cpu",
        action="store_true",
        help="强制 CPU（用于无 GPU 环境测试）",
    )
    return p.parse_args()


def _build_regularizers(cfg: GNNInverterConfig) -> List:
    """
    构造联合反推用的正则列表（不含全局 sum=1 惩罚，避免干扰 testenv/coldway）。

    参数:
        cfg: GNNInverterConfig，读取 lambda_smooth/sparse/anchor。

    返回:
        Regularizer 实例列表。
    """
    return [
        SmoothnessRegularizer(cfg.lambda_smooth),
        SparsityRegularizer(cfg.lambda_sparse),
        AnchorRegularizer(cfg.lambda_anchor),
    ]


def main() -> None:
    """
    CLI 主流程：加载数据与模型 → 构建投影与反推器 → multistart 优化 →
    保存 x_inv.pt、inversion_summary.json、inversion_summary.txt。
    """
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu" if args.force_cpu else _resolve_device(args.device)
    logger.info("使用设备: %s", device)

    x, ys, fs, train_mask, val_mask = load_graph_bundle(args.data_dir)
    if x.shape[1] != INPUT_DIM:
        raise ValueError(f"Expected in_dim={INPUT_DIM}, got {x.shape[1]}")

    graph = torch.load(args.data_dir / "material_graph.pt", map_location="cpu", weights_only=False)
    edge_index, edge_type = merge_hetero_edges(graph)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)

    model, _ = load_dual_rgat(args.ckpt, args.rgat_dir, device)

    bounds = bounds_from_train_x(x, train_mask, margin=args.margin)
    if args.testenv_stats.is_file() and any(
        v is not None for v in (args.tem_lower, args.tem_upper, args.fcr_lower, args.fcr_upper)
    ):
        bounds = bounds_with_physical_testenv(
            x,
            train_mask,
            args.testenv_stats,
            tem_lower=args.tem_lower,
            tem_upper=args.tem_upper,
            fcr_lower=args.fcr_lower,
            fcr_upper=args.fcr_upper,
            margin=args.margin,
        )

    projector = build_projector(x, bounds, total_wt=args.total_wt)
    anchor = x.clone()

    target_ys, target_fs = ys, fs
    if args.target_mode == "model_forward":
        with torch.no_grad():
            target_ys, target_fs = model(x.to(device), edge_index, edge_type)
            target_ys = target_ys.cpu()
            target_fs = target_fs.cpu()

    cfg = GNNInverterConfig(
        optimizer="adam",
        lr=args.lr,
        max_iters=args.max_iters,
        patience=args.patience,
        lambda_smooth=args.lambda_smooth,
        lambda_sparse=1e-5,
        lambda_anchor=args.lambda_anchor,
        lambda_nonneg=0.0,
        lambda_sum1=0.0,
        projection_interval=1,
        projectors=[],
        recon_tol=args.recon_tol,
        device=device,
    )

    inverter = GNNInverter(
        model=model,
        config=cfg,
        regularizers=_build_regularizers(cfg),
        projector=projector,
        anchor=anchor,
    )

    if args.inits.strip():
        names = [s.strip() for s in args.inits.split(",") if s.strip()]
        registry = {
            "training_mean": TrainingMeanInitializer(cfg.restart_noise_scale),
            "dirichlet": DirichletInitializer(alpha=1.0),
            "random_normal": RandomNormalInitializer(scale=0.2),
            "zero": ZeroInitializer(),
        }
        initializers = {n: registry[n] for n in names if n in registry}
        if not initializers:
            raise ValueError(f"No valid inits in {names}")
    elif device == "cpu":
        logger.warning("CPU 模式默认仅 training_mean 初始化（降低内存占用）")
        initializers = {"training_mean": TrainingMeanInitializer(cfg.restart_noise_scale)}
    else:
        initializers = None

    logger.info(
        "开始全图联合反推: Ti余量 A 模式 T_total=%.1f; coldway/testenv=训练分位数 box",
        args.total_wt,
    )
    result = inverter.invert_multistart(
        target_ys=target_ys,
        target_fs=target_fs,
        edge_index=edge_index,
        edge_type=edge_type,
        initializers=initializers,
    )

    x_inv = result.x_inv
    ti_inv = compute_ti_balance(x_inv, args.total_wt)
    ti_true = compute_ti_balance(x, args.total_wt)

    if args.node_mask == "val":
        eval_mask = val_mask
    elif args.node_mask == "train":
        eval_mask = train_mask
    else:
        eval_mask = torch.ones(x.shape[0], dtype=torch.bool)

    with torch.no_grad():
        ys_p, fs_p = model(x_inv.to(device), edge_index, edge_type)
        m = eval_mask
        elem_mae = (x_inv[m, :10] - x[m, :10]).abs().mean().item()
        te_mae = (x_inv[m, 10:12] - x[m, 10:12]).abs().mean().item()
        cw_mae = (x_inv[m, 12:30] - x[m, 12:30]).abs().mean().item()
        ti_mae = (ti_inv[m] - ti_true[m]).abs().mean().item()

    out_pt = args.out_dir / "x_inv.pt"
    torch.save(
        {
            "x_inv": x_inv,
            "x_true": x,
            "ti_balance_inv": ti_inv,
            "ti_balance_true": ti_true,
            "element_names": ELEMENT_NAMES,
            "total_wt": args.total_wt,
            "ti_formula": "Ti(wt%) = total_wt - sum(element_0..9)",
            "target_ys": target_ys,
            "target_fs": target_fs,
            "ys_pred": ys_p.cpu(),
            "fs_pred": fs_p.cpu(),
            "bounds": {
                "testenv_lower": bounds.testenv_lower.tolist(),
                "testenv_upper": bounds.testenv_upper.tolist(),
                "coldway_lower": bounds.coldway_lower.tolist(),
                "coldway_upper": bounds.coldway_upper.tolist(),
            },
            "result": {
                "converged": result.converged,
                "final_recon_mse": result.final_recon_mse,
                "final_ys_mae": result.final_ys_mae,
                "final_fs_mae": result.final_fs_mae,
                "n_iters": result.n_iters,
                "init_name": result.init_name,
            },
            "device": device,
            "target_mode": args.target_mode,
            "format_note": "x_inv 与 material_graph sample.x 同格式；testenv 为 z-score",
        },
        out_pt,
    )

    summary_json_path = args.out_dir / "inversion_summary.json"
    summary_txt_path = args.out_dir / "inversion_summary.txt"
    bounds_dict = {
        "testenv_lower": bounds.testenv_lower.tolist(),
        "testenv_upper": bounds.testenv_upper.tolist(),
        "coldway_lower": bounds.coldway_lower.tolist(),
        "coldway_upper": bounds.coldway_upper.tolist(),
    }
    feature_mae = {
        "element_0_9": elem_mae,
        "ti_balance": ti_mae,
        "testenv_z": te_mae,
        "coldway": cw_mae,
    }
    summary = build_summary_dict(
        device=device,
        total_wt=args.total_wt,
        target_mode=args.target_mode,
        result_converged=result.converged,
        final_recon_mse=result.final_recon_mse,
        final_ys_mae=result.final_ys_mae,
        final_fs_mae=result.final_fs_mae,
        n_iters=result.n_iters,
        best_init=result.init_name,
        eval_mask_name=args.node_mask,
        num_nodes=int(x.shape[0]),
        num_eval_nodes=int(eval_mask.sum().item()),
        feature_mae=feature_mae,
        bounds=bounds_dict,
        paths={
            "x_inv_pt": str(out_pt.resolve()),
            "summary_json": str(summary_json_path.resolve()),
            "summary_txt": str(summary_txt_path.resolve()),
        },
        x_inv=x_inv,
        x_true=x,
        ti_inv=ti_inv,
        ti_true=ti_true,
        ckpt_path=str(args.ckpt.resolve()),
        data_dir=str(args.data_dir.resolve()),
    )
    write_summary_json(summary_json_path, summary)
    sample_idx = torch.where(eval_mask)[0][:3].tolist()
    if len(sample_idx) < 3:
        sample_idx = list(range(min(3, x.shape[0])))
    write_summary_txt(
        summary_txt_path,
        summary,
        x_inv=x_inv,
        x_true=x,
        ti_inv=ti_inv,
        ti_true=ti_true,
        sample_node_indices=sample_idx,
    )

    logger.info(
        "反推完成: recon_mse=%.6f ys_mae=%.6f fs_mae=%.6f ti_mae=%.4f",
        result.final_recon_mse,
        result.final_ys_mae,
        result.final_fs_mae,
        ti_mae,
    )
    logger.info("已保存: %s , %s , %s", out_pt, summary_json_path, summary_txt_path)


if __name__ == "__main__":
    main()
