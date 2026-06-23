"""
ga_virtual_log.py — 每代新增虚拟节点调研日志（JSONL）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from Pareto.ga_archive import ArchiveEntry, weighted_score
from Pareto.ga_report import OutputRestoreContext
from preprocess.preprocess_datagnn_repro import denormalize_targets_to_data1123


def entry_to_virtual_log_record(
    entry: ArchiveEntry,
    restore: OutputRestoreContext,
) -> Dict[str, Any]:
    ys_pred, fs_pred = denormalize_targets_to_data1123(
        entry.fitness.ys_pred,
        entry.fitness.fs_pred,
        ys_mean=restore.ys_mean,
        fs_mean=restore.fs_mean,
    )
    f1_phys = max(0.0, restore.target_ys_physical - ys_pred)
    f2_phys = max(0.0, restore.target_fs_physical - fs_pred)
    return {
        "generation": entry.generation,
        "virtual_id": entry.virtual_id,
        "offspring_kind": entry.offspring_kind,
        "is_immigrant": entry.is_immigrant,
        "immigrant_source": entry.immigrant_source,
        "gene_source": entry.source_label(),
        "weighted_score": weighted_score(entry.fitness),
        "f1_model": entry.fitness.f1,
        "f2_model": entry.fitness.f2,
        "f3_model": entry.fitness.f3,
        "f1_ys_shortfall_phys": f1_phys,
        "f2_fs_shortfall_phys": f2_phys,
        "ys_pred_data1123": ys_pred,
        "fs_pred_data1123": fs_pred,
        "nearest_train_idx": entry.fitness.nearest_train_idx,
        "genome_30d_model": entry.genome.detach().cpu().tolist(),
    }


def append_generation_virtual_log(
    path: Path,
    generation: int,
    new_entries: List[ArchiveEntry],
    restore: OutputRestoreContext,
    *,
    breeding_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """追加本代新增虚拟节点记录（每行一条 JSON）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        if breeding_summary is not None:
            header = {
                "record_type": "generation_header",
                "generation": generation,
                **breeding_summary,
            }
            f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for entry in new_entries:
            rec = entry_to_virtual_log_record(entry, restore)
            rec["record_type"] = "virtual_node"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
