"""
ga_graph.py — 604+1 增广图：背景缓存、新–旧动态边、拼特征矩阵。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from grd.feature_layout import COLDWAY_DIM, ELEMENT_DIM, INPUT_DIM, TESTENV_DIM


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def _cosine_sim_rows(query: np.ndarray, bank: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """query (d,), bank (N, d) -> (N,) cosine similarities."""
    q = query.astype(np.float32, copy=False)
    b = bank.astype(np.float32, copy=False)
    qn = np.linalg.norm(q)
    if qn < eps:
        qn = eps
    bn = np.linalg.norm(b, axis=1)
    bn = np.maximum(bn, eps)
    sim = (b @ q) / (bn * qn)
    return sim.astype(np.float32)


def _heat_sim_new_to_old(
    cold_new: np.ndarray,
    cold_old: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    cold_new: (18,), cold_old: (N, 18) -> (N,) heat_sim 与 rgcn_dataloader 一致。
    """
    c_new = cold_new.reshape(3, 6)
    c_old = cold_old.reshape(-1, 3, 6)
    sims = []
    for r in range(3):
        sims.append(_cosine_sim_rows(c_new[r], c_old[:, r, :], eps=eps))
    return np.mean(np.stack(sims, axis=0), axis=0)


@dataclass
class GraphContext:
    """缓存 604 节点背景图与旧边。"""

    x_old: torch.Tensor  # (N, 30) cpu
    edge_index_old: torch.Tensor  # (2, E_old)
    edge_type_old: torch.Tensor  # (E_old,)
    n_old: int

    @classmethod
    def from_tensors(
        cls,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> GraphContext:
        return cls(
            x_old=x.detach().cpu().float(),
            edge_index_old=edge_index.detach().cpu().long(),
            edge_type_old=edge_type.detach().cpu().long(),
            n_old=int(x.shape[0]),
        )


def edges_new_to_old(
    genome: torch.Tensor,
    ctx: GraphContext,
    *,
    element_thr: float = 0.8,
    testenv_thr: float = 0.8,
    coldway_thr: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算设计节点（索引 n_old）与所有旧节点的 comp/env/heat 双向边。

    返回:
        edge_index_new: (2, E_new) 索引空间为增广图 [0..n_old]。
        edge_type_new: (E_new,) 关系 id 0/1/2。
    """
    g = genome.detach().cpu().numpy().astype(np.float32)
    xo = ctx.x_old.numpy()
    n_old = ctx.n_old
    design_idx = n_old

    elem_old = xo[:, :ELEMENT_DIM]
    te_old = xo[:, ELEMENT_DIM : ELEMENT_DIM + TESTENV_DIM]
    cw_old = xo[:, ELEMENT_DIM + TESTENV_DIM : INPUT_DIM]

    comp_sim = _cosine_sim_rows(g[:ELEMENT_DIM], elem_old)
    env_sim = _cosine_sim_rows(
        g[ELEMENT_DIM : ELEMENT_DIM + TESTENV_DIM],
        te_old,
    )
    heat_sim = _heat_sim_new_to_old(
        g[ELEMENT_DIM + TESTENV_DIM : INPUT_DIM],
        cw_old,
    )

    src_list = []
    dst_list = []
    type_list = []
    for rel_id, sim, thr in (
        (0, comp_sim, element_thr),
        (1, env_sim, testenv_thr),
        (2, heat_sim, coldway_thr),
    ):
        hits = np.where(sim > float(thr))[0]
        for j in hits.tolist():
            src_list.extend([design_idx, int(j)])
            dst_list.extend([int(j), design_idx])
            type_list.extend([rel_id, rel_id])

    if not src_list:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0,), dtype=torch.long),
        )
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type = torch.tensor(type_list, dtype=torch.long)
    return edge_index, edge_type


def build_augmented_graph(
    genome: torch.Tensor,
    ctx: GraphContext,
    *,
    element_thr: float = 0.8,
    testenv_thr: float = 0.8,
    coldway_thr: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    拼增广图 X (N+1, 30) 与合并边。

    返回:
        X, edge_index, edge_type, design_node_idx
    """
    g = genome.detach().cpu().float().reshape(1, INPUT_DIM)
    x_aug = torch.cat([ctx.x_old, g], dim=0)
    ei_new, et_new = edges_new_to_old(
        g.squeeze(0),
        ctx,
        element_thr=element_thr,
        testenv_thr=testenv_thr,
        coldway_thr=coldway_thr,
    )
    if ei_new.numel() == 0:
        edge_index = ctx.edge_index_old.clone()
        edge_type = ctx.edge_type_old.clone()
    else:
        edge_index = torch.cat([ctx.edge_index_old, ei_new], dim=1)
        edge_type = torch.cat([ctx.edge_type_old, et_new], dim=0)
    return x_aug, edge_index, edge_type, ctx.n_old
