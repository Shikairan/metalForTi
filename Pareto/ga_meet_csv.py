"""
ga_meet_csv.py — 达标基因（f1=f2=0）去重写入 CSV（默认关闭，由 CLI 开启）。
每代收集后立即落盘，避免长跑仅内存累积、中途崩溃全丢。
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch

from preprocess.preprocess_datagnn_repro import (
    denormalize_genome_to_data1123,
    denormalize_targets_to_data1123,
)
from Pareto.coldway_display import format_coldway_lines
from Pareto.ga_evaluate import FitnessResult
from Pareto.ga_report import OutputRestoreContext
from grd.feature_layout import ELEMENT_NAMES

logger = logging.getLogger("Pareto.ga_meet_csv")

_EPS = 1e-8
_DEDUP_DECIMALS = 6

CSV_FIELDS = [
    "uts_pred",
    "fs_pred",
    "tem",
    "sr",
    *[f"el_{n}" for n in ELEMENT_NAMES],
    "coldway",
]
# 中文 Windows / Excel 默认按系统 ANSI(GBK) 打开 CSV；用 GBK 避免乱码
CSV_ENCODING = "gbk"


def _genome_dedup_key(genome: torch.Tensor, *, decimals: int = _DEDUP_DECIMALS) -> Tuple[float, ...]:
    g = genome.detach().cpu().float().numpy().reshape(-1)
    return tuple(np.round(g.astype(np.float64), decimals=decimals).tolist())


def meets_target(fit: FitnessResult, *, eps: float = _EPS) -> bool:
    return float(fit.f1) <= eps and float(fit.f2) <= eps


def format_coldway_cell(g_phys_30: np.ndarray) -> str:
    """物理量纲 30 维基因组 → 单格 coldway 可读文本（与日志风格一致）。"""
    cw = np.asarray(g_phys_30, dtype=np.float32).reshape(30)[12:30]
    body = format_coldway_lines(cw)
    lines = ["    【冷加工 coldway】"]
    for line in body:
        lines.append("  " + line)
    return "\n".join(lines)


class MeetTargetCsvCollector:
    """跨代收集达标个体；按基因组去重；每代可立刻原子落盘。"""

    def __init__(
        self,
        restore: OutputRestoreContext,
        path: Optional[Path] = None,
    ) -> None:
        self.restore = restore
        self.path = Path(path) if path is not None else None
        self._seen: Set[Tuple[float, ...]] = set()
        self._rows: List[Dict[str, Any]] = []
        if self.path is not None:
            self.flush(force=True)  # 启动时写出表头，目录即可见文件

    def __len__(self) -> int:
        return len(self._rows)

    def add_individuals(
        self,
        individuals: Iterable[Any],
        *,
        generation: int,
        flush: bool = True,
    ) -> int:
        """加入本轮达标个体；有新增时立刻落盘。返回本轮新条目数。"""
        del generation  # 不写入 CSV，保留参数以兼容调用方
        n_new = 0
        for ind in individuals:
            fit = getattr(ind, "fitness", None)
            genome = getattr(ind, "genome", None)
            if fit is None or genome is None:
                continue
            if not meets_target(fit):
                continue
            key = _genome_dedup_key(genome)
            if key in self._seen:
                continue
            self._seen.add(key)
            self._rows.append(self._row_from(genome, fit))
            n_new += 1
        if flush and self.path is not None and n_new > 0:
            self.flush()
        return n_new

    def _row_from(
        self,
        genome: torch.Tensor,
        fit: FitnessResult,
    ) -> Dict[str, Any]:
        g_model = genome.detach().cpu().float()
        g_phys = denormalize_genome_to_data1123(
            g_model.numpy(),
            te_mean=self.restore.te_mean,
            te_std=self.restore.te_std,
        )
        uts_phys, fs_phys = denormalize_targets_to_data1123(
            fit.ys_pred,
            fit.fs_pred,
            ys_mean=self.restore.ys_mean,
            fs_mean=self.restore.fs_mean,
        )
        row: Dict[str, Any] = {
            "uts_pred": float(uts_phys),
            "fs_pred": float(fs_phys),
            "tem": float(g_phys[10]),
            "sr": float(g_phys[11]),
        }
        for i, name in enumerate(ELEMENT_NAMES):
            row[f"el_{name}"] = float(g_phys[i])
        row["coldway"] = format_coldway_cell(g_phys)
        return row

    def flush(self, *, force: bool = False) -> None:
        """原子覆盖写入 CSV（先写 .tmp 再 replace）。"""
        if self.path is None:
            if force:
                raise ValueError("MeetTargetCsvCollector.path 未设置，无法 flush")
            return
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding=CSV_ENCODING) as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
            w.writeheader()
            for row in self._rows:
                w.writerow(row)
            f.flush()
        tmp.replace(path)

    def write_csv(self, path: Optional[Path] = None) -> None:
        """兼容旧接口：写入指定 path 或 self.path。"""
        if path is not None:
            self.path = Path(path)
        self.flush(force=True)
        logger.info(
            "达标基因 CSV 已写入 %s（去重后 %d 条，编码 %s）",
            self.path.resolve() if self.path else "(none)",
            len(self._rows),
            CSV_ENCODING,
        )
