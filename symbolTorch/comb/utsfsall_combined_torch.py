#!/usr/bin/env python3
"""
可直接使用的 PyTorch 无图公式模块：utsFsAll 组合预测 UTS + FS。

来源 run: comb/runs/full_utsFsAll_it400_ms40
  Y = L_ridge(x) + R_symbolic(x)

输入 x: float tensor [..., 30]，列顺序与 datagnnUts.csv / FEATURE_NAMES 一致：
  Al, Zr, Sn, Mo, Cr, Nb, el_Si, V, Ta, Fe,
  tem, fcr,
  coldway_0 .. coldway_17

输出为模型量纲（与 uts.pt / fs.pt 一致）。物理量可乘 mean_UTS / mean_FS。

示例:
  from comb.utsfsall_combined_torch import UtsFsAllCombinedFormula, load_from_run_dir
  model = load_from_run_dir()
  x = torch.randn(8, 30)
  uts, fs = model(x)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

# 默认公式产物路径（相对本文件）
_DEFAULT_RUN_DIR = Path(__file__).resolve().parent / "runs" / "full_utsFsAll_it400_ms40"
_DEFAULT_EQUATIONS_JSON = _DEFAULT_RUN_DIR / "final_equations.json"

# 列索引（与 common.constants.FEATURE_NAMES 一致）
IDX = {
    "Al": 0,
    "Zr": 1,
    "Sn": 2,
    "Mo": 3,
    "Cr": 4,
    "Nb": 5,
    "el_Si": 6,
    "V": 7,
    "Ta": 8,
    "Fe": 9,
    "tem": 10,
    "fcr": 11,
}
IDX_COLDWAY_0 = 12


@dataclass
class FormulaOutputs:
    """forward(..., return_parts=True) 时的分解输出。"""

    uts: torch.Tensor
    fs: torch.Tensor
    linear_uts: torch.Tensor
    linear_fs: torch.Tensor
    residual_uts: torch.Tensor
    residual_fs: torch.Tensor


def _load_heads_from_json(path: Path) -> Tuple[dict, dict, dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    heads = payload["heads"]
    codec = payload.get("codec", {})
    uts = heads["UTS"]
    fs = heads["FS"]
    return uts, fs, codec


class UtsFsAllCombinedFormula(nn.Module):
    """
    固定系数 + 符号残差的组合公式（无可训练参数）。

    forward(x) -> (uts, fs)
    """

    def __init__(
        self,
        *,
        uts_intercept: float,
        uts_coef: torch.Tensor,
        fs_intercept: float,
        fs_coef: torch.Tensor,
        mean_uts: float = 1291.9939771048014,
        mean_fs: float = 28.120464644701983,
    ) -> None:
        super().__init__()
        self.register_buffer("uts_coef", uts_coef.reshape(-1).to(dtype=torch.float64))
        self.register_buffer("fs_coef", fs_coef.reshape(-1).to(dtype=torch.float64))
        self.register_buffer(
            "uts_intercept", torch.tensor(float(uts_intercept), dtype=torch.float64)
        )
        self.register_buffer(
            "fs_intercept", torch.tensor(float(fs_intercept), dtype=torch.float64)
        )
        self.register_buffer("mean_uts", torch.tensor(float(mean_uts), dtype=torch.float64))
        self.register_buffer("mean_fs", torch.tensor(float(mean_fs), dtype=torch.float64))

    @classmethod
    def from_equations_json(cls, path: Union[str, Path]) -> "UtsFsAllCombinedFormula":
        uts_h, fs_h, codec = _load_heads_from_json(Path(path))
        return cls(
            uts_intercept=float(uts_h["linear_intercept"]),
            uts_coef=torch.tensor(uts_h["linear_coefficients"], dtype=torch.float64),
            fs_intercept=float(fs_h["linear_intercept"]),
            fs_coef=torch.tensor(fs_h["linear_coefficients"], dtype=torch.float64),
            mean_uts=float(codec.get("mean_UTS", 1291.9939771048014)),
            mean_fs=float(codec.get("mean_FS_dataOri2", 28.120464644701983)),
        )

    @staticmethod
    def _as_2d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return x.unsqueeze(0)
        if x.ndim != 2 or x.shape[-1] != 30:
            raise ValueError(f"expected [..., 30], got shape {tuple(x.shape)}")
        return x

    def linear_uts(self, x: torch.Tensor) -> torch.Tensor:
        x = self._as_2d(x).to(dtype=self.uts_coef.dtype, device=self.uts_coef.device)
        return self.uts_intercept + x @ self.uts_coef

    def linear_fs(self, x: torch.Tensor) -> torch.Tensor:
        x = self._as_2d(x).to(dtype=self.fs_coef.dtype, device=self.fs_coef.device)
        return self.fs_intercept + x @ self.fs_coef

    def residual_uts(self, x: torch.Tensor) -> torch.Tensor:
        x = self._as_2d(x).to(dtype=self.uts_coef.dtype, device=self.uts_coef.device)
        al = x[:, IDX["Al"]]
        zr = x[:, IDX["Zr"]]
        mo = x[:, IDX["Mo"]]
        si = x[:, IDX["el_Si"]]
        tem = x[:, IDX["tem"]]
        fcr = x[:, IDX["fcr"]]
        # R_UTS = fcr*tem*(fcr/(Al+0.08873339))*(0.10387048 - 0.097857766*Zr)*(Mo+el_Si)
        return (
            fcr
            * tem
            * (fcr / (al + 0.08873339))
            * (0.10387048 - 0.097857766 * zr)
            * (mo + si)
        )

    def residual_fs(self, x: torch.Tensor) -> torch.Tensor:
        x = self._as_2d(x).to(dtype=self.fs_coef.dtype, device=self.fs_coef.device)
        al = x[:, IDX["Al"]]
        tem = x[:, IDX["tem"]]
        fcr = x[:, IDX["fcr"]]
        # fcr - (-Al - 1.2845563) = fcr + Al + 1.2845563
        denom = fcr * (fcr + al + 1.2845563)
        inner = tem + 0.65531564 / denom
        return (tem - 0.5413826) * inner * 0.3372095 - 0.2900931

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_parts: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], FormulaOutputs]:
        """
        Parameters
        ----------
        x : [N, 30] 或 [30]
        return_parts : 若 True，返回 FormulaOutputs（含线性/残差分解）

        Returns
        -------
        uts, fs : 模型量纲，形状 [N]
        """
        squeeze = x.ndim == 1
        l_uts = self.linear_uts(x)
        l_fs = self.linear_fs(x)
        r_uts = self.residual_uts(x)
        r_fs = self.residual_fs(x)
        uts = l_uts + r_uts
        fs = l_fs + r_fs
        if squeeze:
            uts = uts.squeeze(0)
            fs = fs.squeeze(0)
            l_uts = l_uts.squeeze(0)
            l_fs = l_fs.squeeze(0)
            r_uts = r_uts.squeeze(0)
            r_fs = r_fs.squeeze(0)
        if return_parts:
            return FormulaOutputs(
                uts=uts, fs=fs,
                linear_uts=l_uts, linear_fs=l_fs,
                residual_uts=r_uts, residual_fs=r_fs,
            )
        return uts, fs

    @torch.no_grad()
    def to_physical(
        self,
        uts_model: torch.Tensor,
        fs_model: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """模型量纲 -> UTS_MPa, FS_dataOri2。"""
        return uts_model * self.mean_uts, fs_model * self.mean_fs


def load_from_run_dir(
    run_dir: Optional[Union[str, Path]] = None,
    *,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> UtsFsAllCombinedFormula:
    """从 comb run 目录加载（默认 full_utsFsAll_it400_ms40）。"""
    run_dir = Path(run_dir) if run_dir is not None else _DEFAULT_RUN_DIR
    json_path = run_dir / "final_equations.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"missing {json_path}")
    model = UtsFsAllCombinedFormula.from_equations_json(json_path)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def predict_from_csv(
    csv_path: Union[str, Path],
    *,
    model: Optional[UtsFsAllCombinedFormula] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """读 datagnnUts 风格 CSV 前 30 列并预测 UTS/FS（模型量纲）。"""
    import pandas as pd

    df = pd.read_csv(csv_path)
    x = torch.tensor(df.iloc[:, :30].to_numpy(dtype="float64"), device=device, dtype=dtype)
    m = model or load_from_run_dir(device=device, dtype=dtype)
    return m(x)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="utsFsAll combined formula (PyTorch)")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=_DEFAULT_RUN_DIR,
        help="comb run with final_equations.json",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[1].parent
        / "gnnDir"
        / "datacsv"
        / "datagnnUts.csv",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--check", action="store_true", help="对比 per_sample_errors.csv")
    args = p.parse_args()

    model = load_from_run_dir(args.run_dir, device=args.device)
    uts, fs = predict_from_csv(args.csv, model=model, device=args.device)
    print(f"N={uts.shape[0]}  UTS mean={uts.mean().item():.4f}  FS mean={fs.mean().item():.4f}")

    if args.check:
        err_csv = args.run_dir / "per_sample_errors.csv"
        if err_csv.is_file():
            import pandas as pd

            ref = pd.read_csv(err_csv)
            uts_ref = torch.tensor(ref["combined_UTS"].values, device=uts.device, dtype=uts.dtype)
            fs_ref = torch.tensor(ref["combined_FS"].values, device=fs.device, dtype=fs.dtype)
            mae_u = (uts - uts_ref).abs().mean().item()
            mae_f = (fs - fs_ref).abs().mean().item()
            print(f"check vs per_sample_errors.csv  MAE UTS={mae_u:.2e}  FS={mae_f:.2e}")
        else:
            print(f"skip check: {err_csv} not found")
