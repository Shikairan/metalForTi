# dataOri2.csv → datagnn.csv 预处理与逆变换规范

本文档描述将 `symtest/dataOri2.csv`（或结构相同的 `dataOri.csv`）转换为 GNN 建模表 `datagnn.csv` 的**全部数值变换**，并提供**可逆公式**与**可执行复现代码**。按本文操作可独立复现正向预处理与标签/环境的逆变换。

实现参考：

- 正向管线：[`build_datagnn.py`](build_datagnn.py)
- 冷加工核心：[`../pt_dataset.py`](../pt_dataset.py) 中 `_normalize_coldway_tx_cx`、`_coldway_row_to_seq_3x6`

---

## 1. 数据约定

| 项目 | 说明 |
|------|------|
| 输入 | `dataOri2.csv`，604 行数据行（不含表头） |
| 输出 | `datagnn.csv`，604 行 × 32 列 |
| 行顺序 | **严格保持**，第 `i` 行输入对应第 `i` 行输出（GNN 节点索引） |
| 忽略列 | `UTS`、`imgf` 及末尾无名附加列**不参与**变换 |
| dtype | 全程 `float32` 计算与存储（统计量可用 `float64` 求取后写入） |

### 1.1 输入列 → 输出列映射

| 输出 | 输入列名 | 维数 |
|------|----------|------|
| `element_0` … `element_9` | Al, Zr, Sn, Mo, Cr, Nb, Si, V, Ta, Fe | 10 |
| `testenv_0`, `testenv_1` | tem, fcr | 2 |
| `coldway_0` … `coldway_17` | T1,t1,T2,t2,T3,t3,C1_1…C3_3 | 18（由 15 列导出） |
| `YS`, `FS` | YS, FS | 2 |

输出列顺序：

```
element_0..9 | testenv_0..1 | coldway_0..17 | YS | FS
     10      |      2       |      18       | 2  |  = 32
```

---

## 2. 全局统计常量（基于 dataOri2.csv 全表 604 行）

以下常量用于 **testenv 标准化** 与 **YS/FS 均值归一化**。若输入 CSV 行集变化，必须**重新计算**；否则逆变换将产生系统误差。

### 2.1 testenv（Z-score）

| 列 | mean μ | std σ（ddof=0；σ=0 时替换为 1） |
|----|--------|----------------------------------|
| tem | 194.7433774834437 | 214.62536752046483 |
| fcr | 1342.7152317880796 | 1547.0319516107018 |

持久化文件：与 `datagnn.csv` 同目录的 `testenv_stats.csv`（列：`col, mean, std`）。

### 2.2 标签均值归一化

| 列 | mean（全表算术平均） |
|----|----------------------|
| YS | 965.7821034430465 |
| FS | 28.120464644701983 |

小常数：`eps = 1e-8`（仅用于分母稳定，防止除零）。

---

## 3. 分块变换

### 3.1 Element（10 维）— 恒等变换 `nofix`

**正向：**

\[
\text{element}_j = \text{raw}_j,\quad j=0..9
\]

**逆向：**

\[
\text{raw}_j = \text{element}_j
\]

无额外参数，完全可逆。

---

### 3.2 Testenv（2 维）— 列向 Z-score

**正向（第 j 列，j∈{0:tem, 1:fcr}）：**

\[
z_j = \frac{x_j - \mu_j}{\sigma_j'}
\quad\text{其中}\quad
\sigma_j' = \begin{cases} \sigma_j & \sigma_j \neq 0 \\ 1 & \sigma_j = 0 \end{cases}
\]

- \(\mu_j, \sigma_j\) 为**全表**该列均值与标准差（`numpy.std(..., ddof=0)`）。

**逆向：**

\[
x_j = z_j \cdot \sigma_j' + \mu_j
\]

完全可逆（在浮点精度内）。

**示例（第 1 行 tem=25, fcr=0）：**

```
testenv_0 = (25 - 194.7433774834437) / 214.62536752046483 ≈ -0.79088324
testenv_1 = (0 - 1342.7152317880796) / 1547.0319516107018 ≈ -0.86793023
```

---

### 3.3 Coldway（15 列 → 18 维）

冷加工分两步：**Tx 对数缩放** + **3×6 序列展平**。

#### 3.3.1 原始 15 列索引

```
idx:  0    1    2    3    4    5    6     7     8     9    10    11    12    13    14
col: T1   t1   T2   t2   T3   t3  C1_1  C1_2  C1_3  C2_1  C2_2  C2_3  C3_1  C3_2  C3_3
```

占位符：任意原始值为 **-1** 表示该字段无效/未使用。

#### 3.3.2 步骤 A：Tx 缩放 `_normalize_coldway_tx_cx`

对每一行 `raw[0..14]` 得到 `scaled[0..14]`：

| 下标 | 列 | 正向规则 |
|------|-----|----------|
| 0, 2, 4 | T1, T2, T3 | 若 `raw[k] ≠ -1`：`scaled[k] = ln(raw[k] / 800)`；否则 `scaled[k] = -1` |
| 1, 3, 5 | t1, t2, t3 | 若 `raw[k] ≠ -1`：`scaled[k] = ln(raw[k])`；否则 `scaled[k] = -1` |
| 6..14 | C* | `scaled[k] = raw[k]`（0 / 1 / -1 不变） |

固定除数：`T_div = 800`（温度），`t_div = 1`（时间，即直接取自然对数）。

**逆向（仅对 Tx，在已知非 -1 时）：**

```
T_raw = 800 * exp(T_scaled)    # 对 T1,T2,T3
t_raw = exp(t_scaled)          # 对 t1,t2,t3
```

C 列在缩放步为恒等，逆变换亦恒等。

#### 3.3.3 步骤 B：构造 3×6 并展平 `_coldway_row_to_seq_3x6`

**C 矩阵（3×3，由原始行下标 6..14 按行填入）：**

```
C[0,:] = [C1_1, C1_2, C1_3]   # 工艺组 1，配对 (T1, t1)
C[1,:] = [C2_1, C2_2, C2_3]   # 工艺组 2，配对 (T2, t2)
C[2,:] = [C3_1, C3_2, C3_3]   # 工艺组 3，配对 (T3, t3)
```

**中间张量 `cell` 形状 (3, 3, 2)**，对每个 `i ∈ {0,1,2}`（时间步）、`j ∈ {0,1,2}`（槽位）：

```
T_raw_i, t_raw_i = raw 中第 i 组的温度/时间（T1/t1 或 T2/t2 或 T3/t3）
invalid_t = (T_raw_i == -1) or (t_raw_i == -1)

若 C[i,j] == 1 且 not invalid_t:
    cell[i,j] = [scaled_T_i, scaled_t_i]   # 来自步骤 A 的 scaled
否则:
    cell[i,j] = [0, 0]
```

**展平为 18 维：**

```
seq = cell.reshape(3, 6)   # C 顺序：行 i 的 6 个数 = [cell[i,0,0], cell[i,0,1], cell[i,1,0], cell[i,1,1], cell[i,2,0], cell[i,2,1]]
coldway_0..17 = seq.reshape(-1)
```

**示例（T1=1020, t1=1, C1_1=1，其余组为 -1）：**

```
scaled_T1 = ln(1020/800) ≈ 0.24294616
coldway_0 = 0.24294616，coldway_1..17 = 0
```

#### 3.3.4 Coldway 逆变换（部分可逆）

> **注意 1：** 全零的 coldway 向量可能来自「整组 -1」或「C 全为 0」，逆变换无法唯一还原原始 -1 占位。

> **注意 2（信息损失）：** 当某激活槽位满足 `T_raw=800` 且 `t_raw=1` 时，正向缩放得到 `T_scaled=ln(800/800)=0`、`t_scaled=ln(1)=0`，写入 `cell[i,j]=[0,0]`，与**未激活槽位**完全相同。`dataOri2.csv` 中约有 **37 个**此类槽位，**无法仅从 18 维 coldway 唯一逆推**原始 T/t/C。element / testenv / YS / FS 的逆变换不受此影响。

**步骤：**

1. `seq = coldway.reshape(3, 6)`，再 `cell = seq.reshape(3, 3, 2)`。
2. 对每个 `(i,j)`：
   - 若 `cell[i,j,0] != 0` 或 `cell[i,j,1] != 0`：
     - `C[i,j] = 1`
     - `T_scaled = cell[i,j,0]`, `t_scaled = cell[i,j,1]`
     - `T_raw = 800 * exp(T_scaled)`, `t_raw = exp(t_scaled)`
     - 写回对应组的 T/t 列（组 0→T1/t1，组 1→T2/t2，组 2→T3/t3）
   - 否则：`C[i,j] = 0`；若该组其它槽位亦无数据，可将该组 T/t 设为 -1。
3. 若业务上已知「T=800, t=1 且 C 某列为 1」的样本，需保留原始 CSV 或在逆变换时合并外部元数据。

特征向量 → 原始 15 列：对**非零 cell 条目**可精确 `exp` 还原；对全零槽位存在上述歧义。

---

### 3.4 标签 YS / FS — 列均值归一化（mean）

仓库中现有 `datagnn.csv` 的标签经此步处理。`dataOri2.csv` 中 YS/FS 为**物理量**；`dataOri.csv` 中已为归一化值。

**正向：**

\[
YS_{out} = \frac{YS_{raw}}{\overline{YS} + \varepsilon},\qquad
FS_{out} = \frac{FS_{raw}}{\overline{FS} + \varepsilon}
\]

其中 \(\overline{YS}\)、\(\overline{FS}\) 为**当前输入表全表**算术均值，\(\varepsilon = 10^{-8}\)。

**逆向：**

\[
YS_{raw} = YS_{out} \cdot (\overline{YS} + \varepsilon),\qquad
FS_{raw} = FS_{out} \cdot (\overline{FS} + \varepsilon)
\]

完全可逆（浮点精度内）。

**示例（第 1 行）：**

```
YS_raw = 1014.805  →  YS_out = 1014.805 / 965.7821034430465 ≈ 1.05075979
FS_raw = 14.697    →  FS_out = 14.697 / 28.120464644701983 ≈ 0.52264428
```

> **与 `build_datagnn.py` 的差异：** 当前脚本写出**未归一化**的 YS/FS。要复现仓库内 `datagnn.csv`，必须在写出前对标签执行本节变换，或直接使用 `dataOri.csv`（其 YS/FS 已归一化）。

---

## 4. 正向预处理总流程（伪代码）

```
输入: dataOri2 第 i 行

1. element_out[0:10]  = [Al, Zr, Sn, Mo, Cr, Nb, Si, V, Ta, Fe]           # nofix

2. testenv_out[0] = (tem - μ_tem) / σ_tem
   testenv_out[1] = (fcr - μ_fcr) / σ_fcr

3. scaled = normalize_coldway_tx_cx(raw_coldway_15)
   coldway_out[0:18] = coldway_row_to_seq_3x6(raw, scaled).flatten()

4. YS_out = YS_raw / (mean(YS_column) + eps)
   FS_out = FS_raw / (mean(FS_column) + eps)

5. 写出 datagnn 第 i 行:
   [element_out | testenv_out | coldway_out | YS_out | FS_out]
```

---

## 5. 逆变换总流程（伪代码）

```
输入: datagnn 第 i 行

1. Al..Fe = element_0..9                                    # 恒等

2. tem = testenv_0 * σ_tem + μ_tem
   fcr = testenv_1 * σ_fcr + μ_fcr

3. (T1..C3_3) = inverse_coldway(coldway_0..17)              # 见 3.3.4，部分可逆

4. YS_raw = YS_out * (mean_YS + eps)
   FS_raw = FS_out * (mean_FS + eps)
```

---

## 6. 可执行复现脚本

将下列代码保存为 `preprocess_datagnn_repro.py`（与本文档同目录），即可一键正向/逆向/校验。

```python
#!/usr/bin/env python3
"""复现 dataOri2 <-> datagnn 正向预处理与逆变换。用法见文件末尾。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ---------- 列名常量 ----------
ELEMENT_COLS = ["Al", "Zr", "Sn", "Mo", "Cr", "Nb", "Si", "V", "Ta", "Fe"]
TESTENV_COLS = ["tem", "fcr"]
TARGET_COLS = ["YS", "FS"]
COLDWAY_COLS = [
    "T1", "t1", "T2", "t2", "T3", "t3",
    "C1_1", "C1_2", "C1_3", "C2_1", "C2_2", "C2_3", "C3_1", "C3_2", "C3_3",
]
EPS = 1e-8
T_DIV = 800.0

# dataOri2.csv 全表统计（输入行集变化时需重算）
DEFAULT_TESTENV_MEAN = np.array([194.7433774834437, 1342.7152317880796], dtype=np.float64)
DEFAULT_TESTENV_STD = np.array([214.62536752046483, 1547.0319516107018], dtype=np.float64)
DEFAULT_YS_MEAN = 965.7821034430465
DEFAULT_FS_MEAN = 28.120464644701983


def normalize_coldway_tx_cx(raw: np.ndarray) -> np.ndarray:
    """raw: (15,) or (N,15) -> scaled same shape, float32."""
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
  """raw, scaled: (15,) -> (18,) float32."""
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
  """逆变换：18 -> 15。无法唯一恢复的 -1 占位按「无激活槽则 T/t=-1, C=0」处理。"""
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
        w.writerow(
          te[i].tolist() + el[i].tolist() + cw[i].tolist() + tg[i].tolist()
        )
    print(f"[OK] inverse -> {out_path} rows={el.shape[0]}")

  else:  # verify
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
```

### 6.1 命令示例

```bash
cd symbolTransformer/symtest/allTestPart

# 正向：dataOri2 -> datagnn（含 YS/FS 均值归一化）
python preprocess_datagnn_repro.py forward \
  --input ../dataOri2.csv \
  --output datagnn_repro.csv

# 与仓库 datagnn.csv 逐列校验
python preprocess_datagnn_repro.py verify \
  --input ../dataOri2.csv \
  --reference datagnn.csv

# 逆向：datagnn -> 还原 dataOri 核心列（tem,fcr,元素,冷加工,YS,FS）
python preprocess_datagnn_repro.py inverse \
  --input datagnn.csv \
  --output dataOri_restored.csv
```

---

## 7. 校验清单

| 检查项 | 期望 |
|--------|------|
| 行数 | 604 |
| element | 与输入 Al..Fe 完全一致 |
| testenv | Z-score 后与 §2.1 常量一致 |
| coldway | 与 `pt_dataset` 实现逐元素一致（见 verify） |
| YS/FS | `YS_out = YS_raw / 965.7821034430465`（+eps） |
| 逆变换 testenv | 误差 ≈ 0 |
| 逆变换 YS/FS | 误差 < 1e-10 |
| 逆变换 coldway | 非零 cell 可 `exp` 还原；T=800&t=1 的激活槽（≈37 处）与未激活槽同为 [0,0]，不可唯一逆推 |
| 逆变换 element/testenv/YS/FS | 与原始列最大误差 < 1e-3（float32 量级） |

---

## 8. 附录：dataOri 与 dataOri2 的区别

| 文件 | YS/FS | 特征列 tem..C3_3 |
|------|-------|-------------------|
| `dataOri2.csv` | 物理量（如 1014.805） | 与 dataOri 相同 |
| `dataOri.csv` | 已 mean 归一化（如 1.05076） | 与 dataOri2 相同 |

从 `dataOri2` 生成与仓库一致的 `datagnn.csv`：**必须**对 YS/FS 执行 §3.4；特征三块（element / testenv / coldway）两文件结果相同。

---

*文档版本：与 `build_datagnn.py`、`pt_dataset.py` 当前实现及 `dataOri2.csv`（604 行）统计量对齐。*
