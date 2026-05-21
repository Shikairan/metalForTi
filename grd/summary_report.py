"""
反推结果汇总：生成带中文字段说明的 TXT 与 JSON 报告。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from grd.feature_layout import COLDWAY_DIM, ELEMENT_DIM, ELEMENT_NAMES, INPUT_DIM

# JSON 字段中文说明（写入报告元数据）
FIELD_DESCRIPTIONS_CN: Dict[str, str] = {
    "device": "运行设备（cuda 或 cpu）",
    "total_wt": "钛合金总量标尺（wt%），A 模式下 Ti = total_wt - sum(合金化元素)",
    "target_mode": "反推目标：ground_truth=真实 ys/fs 标签；model_forward=当前 x 的前向预测",
    "converged": "是否达到重建误差容限或早停条件",
    "final_recon_mse": "用反推 x 预测 ys、fs 相对目标的均方误差（越小越好）",
    "final_ys_mae": "屈服强度 YS 预测与目标的平均绝对误差",
    "final_fs_mae": "抗拉/疲劳强度 FS 预测与目标的平均绝对误差",
    "n_iters": "优化实际迭代步数",
    "best_init": "多初始点中重建误差最小的初始化策略名称",
    "eval_mask": "计算特征 MAE 时使用的节点子集（all/train/val）",
    "num_nodes": "图中节点总数",
    "num_eval_nodes": "参与特征 MAE 统计的节点数",
    "feature_mae_on_eval": "反推特征与真值 x 的分段平均绝对误差（在 eval_mask 上）",
    "element_0_9": "10 个显式合金化元素（Al…Fe）的 MAE",
    "ti_balance": "钛余量 Ti(wt%) 的 MAE，Ti=total_wt-sum(element)",
    "testenv_z": "试验环境 tem、fcr 的 z-score MAE（与训练张量同空间）",
    "coldway": "18 维工艺 coldway 特征的 MAE",
    "bounds": "反推投影使用的上下界",
    "testenv_lower": "testenv 维下界（z-score 空间，列顺序 tem, fcr）",
    "testenv_upper": "testenv 维上界（z-score 空间）",
    "coldway_lower": "coldway 18 维下界（训练分位数，与 x 一致）",
    "coldway_upper": "coldway 18 维上界",
    "paths": "输出文件路径",
    "x_inv_pt": "PyTorch 主结果文件，含 x_inv 张量及 ti_balance 等",
    "summary_json": "本 JSON 汇总（机器可读）",
    "summary_txt": "本 TXT 汇总（人类可读，含中文说明）",
    "x_inv_shape": "反推输入特征张量形状 [节点数, 30]",
    "ti_balance_inv_mean": "反推钛余量在全图上的平均值",
    "ti_balance_true_mean": "真值钛余量在全图上的平均值",
    "element_sum_inv_mean": "反推 10 元含量之和的平均值（应 ≤ total_wt）",
    "element_sum_true_mean": "真值 10 元含量之和的平均值",
    "field_descriptions": "各 JSON 字段的中文含义（本字典）",
}


def _fmt_float(v: float, nd: int = 6) -> str:
    if abs(v) >= 1e4 or (abs(v) < 1e-3 and v != 0):
        return f"{v:.4e}"
    return f"{v:.{nd}f}"


def build_summary_dict(
    *,
    device: str,
    total_wt: float,
    target_mode: str,
    result_converged: bool,
    final_recon_mse: float,
    final_ys_mae: float,
    final_fs_mae: float,
    n_iters: int,
    best_init: str,
    eval_mask_name: str,
    num_nodes: int,
    num_eval_nodes: int,
    feature_mae: Dict[str, float],
    bounds: Dict[str, List[float]],
    paths: Dict[str, str],
    x_inv: torch.Tensor,
    x_true: torch.Tensor,
    ti_inv: torch.Tensor,
    ti_true: torch.Tensor,
    ckpt_path: str,
    data_dir: str,
) -> Dict[str, Any]:
    elem_sum_inv = x_inv[:, :ELEMENT_DIM].sum(dim=1)
    elem_sum_true = x_true[:, :ELEMENT_DIM].sum(dim=1)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "total_wt": total_wt,
        "target_mode": target_mode,
        "converged": result_converged,
        "final_recon_mse": final_recon_mse,
        "final_ys_mae": final_ys_mae,
        "final_fs_mae": final_fs_mae,
        "n_iters": n_iters,
        "best_init": best_init,
        "eval_mask": eval_mask_name,
        "num_nodes": num_nodes,
        "num_eval_nodes": num_eval_nodes,
        "feature_mae_on_eval": feature_mae,
        "bounds": bounds,
        "paths": paths,
        "data_dir": data_dir,
        "ckpt": ckpt_path,
        "x_inv_shape": list(x_inv.shape),
        "ti_balance_inv_mean": float(ti_inv.mean().item()),
        "ti_balance_true_mean": float(ti_true.mean().item()),
        "element_sum_inv_mean": float(elem_sum_inv.mean().item()),
        "element_sum_inv_max": float(elem_sum_inv.max().item()),
        "element_sum_true_mean": float(elem_sum_true.mean().item()),
        "field_descriptions": FIELD_DESCRIPTIONS_CN,
    }


def write_summary_json(path: Path, summary: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_summary_txt(
    path: Path,
    summary: Dict[str, Any],
    *,
    x_inv: Optional[torch.Tensor] = None,
    x_true: Optional[torch.Tensor] = None,
    ti_inv: Optional[torch.Tensor] = None,
    ti_true: Optional[torch.Tensor] = None,
    sample_node_indices: Optional[List[int]] = None,
) -> None:
    """写入带中文说明的 TXT 汇总报告。"""
    lines: List[str] = []
    sep = "=" * 72

    lines.append(sep)
    lines.append("GNN 梯度反推结果汇总报告")
    lines.append(sep)
    lines.append("")
    lines.append(f"生成时间（UTC）: {summary.get('generated_at_utc', '')}")
    lines.append("")

    lines.append("【一、运行配置】")
    lines.append(f"  数据目录: {summary.get('data_dir', '')}")
    lines.append(f"    说明: 含 material_graph.pt、ys.pt、fs.pt 的 PT 数据包路径")
    lines.append(f"  模型权重: {summary.get('ckpt', '')}")
    lines.append(f"    说明: 已训练 DualRGAT 检查点路径")
    lines.append(f"  运行设备: {summary.get('device', '')}")
    lines.append(f"    说明: {FIELD_DESCRIPTIONS_CN['device']}")
    lines.append(f"  反推目标模式: {summary.get('target_mode', '')}")
    lines.append(f"    说明: {FIELD_DESCRIPTIONS_CN['target_mode']}")
    lines.append(f"  钛总量标尺 total_wt: {summary.get('total_wt', 100)} wt%")
    lines.append(f"    说明: {FIELD_DESCRIPTIONS_CN['total_wt']}")
    lines.append(f"  评估节点子集 eval_mask: {summary.get('eval_mask', '')}")
    lines.append(f"    说明: {FIELD_DESCRIPTIONS_CN['eval_mask']}")
    lines.append(f"  节点总数: {summary.get('num_nodes', '')}")
    lines.append(f"  参与特征 MAE 的节点数: {summary.get('num_eval_nodes', '')}")
    lines.append("")

    lines.append("【二、优化与重建指标】")
    lines.append(f"  是否收敛 converged: {summary.get('converged', '')}")
    lines.append(f"    说明: {FIELD_DESCRIPTIONS_CN['converged']}")
    lines.append(f"  重建均方误差 final_recon_mse: {_fmt_float(summary.get('final_recon_mse', 0))}")
    lines.append(f"    说明: {FIELD_DESCRIPTIONS_CN['final_recon_mse']}")
    lines.append(f"  YS 平均绝对误差 final_ys_mae: {_fmt_float(summary.get('final_ys_mae', 0))}")
    lines.append(f"    说明: {FIELD_DESCRIPTIONS_CN['final_ys_mae']}")
    lines.append(f"  FS 平均绝对误差 final_fs_mae: {_fmt_float(summary.get('final_fs_mae', 0))}")
    lines.append(f"    说明: {FIELD_DESCRIPTIONS_CN['final_fs_mae']}")
    lines.append(f"  迭代次数 n_iters: {summary.get('n_iters', '')}")
    lines.append(f"    说明: {FIELD_DESCRIPTIONS_CN['n_iters']}")
    lines.append(f"  最优初始化 best_init: {summary.get('best_init', '')}")
    lines.append(f"    说明: {FIELD_DESCRIPTIONS_CN['best_init']}")
    lines.append("")

    lines.append("【三、反推特征与真值对比（分段 MAE）】")
    mae = summary.get("feature_mae_on_eval", {})
    for key, label in [
        ("element_0_9", "合金化元素 element_0..9（Al,Zr,Sn,Mo,Cr,Nb,Si,V,Ta,Fe）"),
        ("ti_balance", "钛余量 Ti(wt%)=total_wt-sum(元素)"),
        ("testenv_z", "试验环境 testenv（tem,fcr 的 z-score）"),
        ("coldway", "工艺 coldway 18 维"),
    ]:
        if key in mae:
            lines.append(f"  {label}")
            lines.append(f"    MAE = {_fmt_float(mae[key])}")
    lines.append("")

    lines.append("【四、钛余量与元素含量统计（全图）】")
    lines.append(f"  反推钛余量均值 ti_balance_inv_mean: {_fmt_float(summary.get('ti_balance_inv_mean', 0), 4)} wt%")
    lines.append(f"  真值钛余量均值 ti_balance_true_mean: {_fmt_float(summary.get('ti_balance_true_mean', 0), 4)} wt%")
    lines.append(f"  反推 10 元含量之和 均值: {_fmt_float(summary.get('element_sum_inv_mean', 0), 4)} wt%")
    lines.append(f"  反推 10 元含量之和 最大值: {_fmt_float(summary.get('element_sum_inv_max', 0), 4)} wt%（应 ≤ total_wt）")
    lines.append(f"  真值 10 元含量之和 均值: {_fmt_float(summary.get('element_sum_true_mean', 0), 4)} wt%")
    lines.append("    说明: Ti 不在 x 向量内，由 100 减去 10 个合金化元素之和得到")
    lines.append("")

    bounds = summary.get("bounds", {})
    lines.append("【五、投影约束上下界】")
    lines.append(f"  说明: 反推每步投影到下列范围内；testenv 为 z-score，与训练 material_graph 一致")
    if "testenv_lower" in bounds:
        lines.append(f"  testenv 下界 (tem, fcr): {bounds['testenv_lower']}")
        lines.append(f"    说明: {FIELD_DESCRIPTIONS_CN['testenv_lower']}")
    if "testenv_upper" in bounds:
        lines.append(f"  testenv 上界 (tem, fcr): {bounds['testenv_upper']}")
        lines.append(f"    说明: {FIELD_DESCRIPTIONS_CN['testenv_upper']}")
    if "coldway_lower" in bounds:
        lo = bounds["coldway_lower"]
        lines.append(f"  coldway 下界 (18 维): 前3维={lo[:3]} … 共{len(lo)}维")
        lines.append(f"    说明: {FIELD_DESCRIPTIONS_CN['coldway_lower']}")
    if "coldway_upper" in bounds:
        hi = bounds["coldway_upper"]
        lines.append(f"  coldway 上界 (18 维): 前3维={hi[:3]} … 共{len(hi)}维")
        lines.append(f"    说明: {FIELD_DESCRIPTIONS_CN['coldway_upper']}")
    lines.append("")

    lines.append("【六、输出文件说明】")
    paths = summary.get("paths", {})
    for key, desc_key in [
        ("x_inv_pt", "x_inv_pt"),
        ("summary_json", "summary_json"),
        ("summary_txt", "summary_txt"),
    ]:
        if key in paths:
            lines.append(f"  {paths[key]}")
            lines.append(f"    说明: {FIELD_DESCRIPTIONS_CN.get(desc_key, paths[key])}")
    lines.append("")
    lines.append("  x_inv.pt 内主要张量:")
    lines.append(f"    x_inv      — 形状 {summary.get('x_inv_shape', [604, 30])}，反推得到的 30 维节点特征")
    lines.append("    x_true     — 原始真值特征（与 material_graph sample.x 相同）")
    lines.append("    ti_balance_inv / ti_balance_true — 钛余量 (N,)，单位 wt%")
    lines.append(f"    element_names — {ELEMENT_NAMES}")
    lines.append("    列布局: [0:10] 元素 | [10:12] testenv z-score | [12:30] coldway")
    lines.append("    target_ys, target_fs — 反推目标标签")
    lines.append("    ys_pred, fs_pred — 用 x_inv 前向 GNN 的预测值")
    lines.append("")

    if (
        x_inv is not None
        and x_true is not None
        and ti_inv is not None
        and ti_true is not None
        and sample_node_indices
    ):
        lines.append("【七、样例节点对比（反推 vs 真值）】")
        lines.append("  说明: 下列为部分节点的含量与钛余量对比，便于抽查")
        for idx in sample_node_indices:
            if idx < 0 or idx >= x_inv.shape[0]:
                continue
            lines.append(f"  --- 节点 {idx} ---")
            inv_e = x_inv[idx, :ELEMENT_DIM].tolist()
            tru_e = x_true[idx, :ELEMENT_DIM].tolist()
            for i, name in enumerate(ELEMENT_NAMES):
                lines.append(
                    f"    {name}: 反推={_fmt_float(inv_e[i], 4)}  真值={_fmt_float(tru_e[i], 4)}"
                )
            lines.append(
                f"    10元之和: 反推={_fmt_float(sum(inv_e), 4)}  真值={_fmt_float(sum(tru_e), 4)}"
            )
            lines.append(
                f"    Ti余量: 反推={_fmt_float(ti_inv[idx].item(), 4)}  "
                f"真值={_fmt_float(ti_true[idx].item(), 4)} wt%"
            )
            lines.append(
                f"    testenv(z): 反推={[_fmt_float(v, 4) for v in x_inv[idx, 10:12].tolist()]}  "
                f"真值={[_fmt_float(v, 4) for v in x_true[idx, 10:12].tolist()]}"
            )
        lines.append("")

    lines.append("【八、JSON 字段中文索引】")
    lines.append("  完整机器可读汇总见 inversion_summary.json 中的 field_descriptions 字段")
    for k, v in FIELD_DESCRIPTIONS_CN.items():
        if k in summary or k in ("element_0_9", "ti_balance", "testenv_z", "coldway"):
            continue
        if k.startswith("testenv") or k.startswith("coldway") or k in ("paths", "x_inv_pt"):
            continue
    for k in sorted(FIELD_DESCRIPTIONS_CN.keys()):
        lines.append(f"    {k}: {FIELD_DESCRIPTIONS_CN[k]}")
    lines.append("")
    lines.append(sep)
    lines.append("报告结束")
    lines.append(sep)

    path.write_text("\n".join(lines), encoding="utf-8")
