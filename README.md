# metalForTi — 钛合金 YS/FS 图神经网络与逆设计

基于材料组分、试验环境与工艺特征，用 **图神经网络（GNN）** 预测钛合金 **YS（屈服强度）** 与 **FS**，并支持在冻结模型上对 **30 维输入特征** 做 **梯度反推** 与 **NSGA-II 帕累托遗传逆设计**。

**主开发分支：`meta4TiiGnn`**

---

## 仓库结构

| 目录 | 作用 | 详细文档 |
|------|------|----------|
| [`preprocess/`](preprocess/) | **data1123 / dataOri2 ↔ datagnn** 正向与逆变换（Pareto/grd 量纲） | [PREPROCESS_datagnn.md](preprocess/PREPROCESS_datagnn.md) |
| [`gnnDir/`](gnnDir/) | 数据构建、异质图 PT、RGAT/R-GCN **训练** | [gnnDir/README.md](gnnDir/README.md) |
| [`modelAll/`](modelAll/) | **全量 604 条** RGAT 训练与默认 checkpoint | [modelAll/README.md](modelAll/README.md) |
| [`grd/`](grd/) | 冻结 GNN 下的 **全特征梯度反推** | [grd/README.md](grd/README.md) |
| [`Pareto/`](Pareto/) | **NSGA-II 帕累托遗传逆设计**（目标 YS/FS → 30 维配方） | [Pareto/README.md](Pareto/README.md) |
| [`symbolTorch/`](symbolTorch/) | GNN 蒸馏为 **可读符号公式**（SymTorch + PySR） | [symbolTorch/README.md](symbolTorch/README.md) |

典型流水线：

```text
dataOri2 / data1123
    → preprocess（特征与量纲）
    → gnnDir 构图/训练 或 modelAll 全量训练
    → best_rgat_full.pt
         ↓
    grd 梯度反推（全图还原验证）
         ↓
    Pareto 遗传逆设计（用户 YS/FS → 帕累托配方前沿）
         ↓
    symbolTorch 符号蒸馏（可选）
```

> **说明**：历史脚本曾依赖根目录 `pt_dataset.py`，已废弃；冷加工与 datagnn 逻辑统一在 [`preprocess/preprocess_datagnn_repro.py`](preprocess/preprocess_datagnn_repro.py)。

---

## Python 版本

| 用途 | 推荐 Python | 说明 |
|------|-------------|------|
| **gnnDir + grd + Pareto + modelAll** | **3.10** | 与 `requirements.txt` 中 `torch==2.0.1` 一致 |
| **symbolTorch** | **≥ 3.11** | `torch-symbolic` 要求；建议单独虚拟环境 |

```bash
python --version   # 主环境建议 3.10.x
```

---

## 安装

### 1. 克隆与分支

```bash
git clone https://github.com/Shikairan/metalForTi.git
cd metalForTi
git checkout meta4TiiGnn
```

### 2. 虚拟环境（Python 3.10）

```bash
conda create -n metal4ti python=3.10 -y
conda activate metal4ti
# 或: python3.10 -m venv .venv && source .venv/bin/activate
```

### 3. PyTorch 与依赖

**GPU（CUDA 11.8 示例）：**

```bash
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**仅 CPU：**

```bash
pip install torch==2.0.1
pip install -r requirements.txt
```

### 4. PYTHONPATH

在 **仓库根目录** 执行：

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

---

## 快速开始

以下命令均在仓库根目录、已 `activate` 且已设置 `PYTHONPATH` 的前提下执行。

### A. 默认资源（通常无需重训）

| 资源 | 默认路径 |
|------|----------|
| 图数据 PT | `gnnDir/gnndataPT/r-gatPT/` |
| **Pareto / 推荐 checkpoint** | `modelAll/runs/best_rgat_full.pt` |
| 备选 checkpoint（r-gatDouble） | `gnnDir/gnn/r-gatDouble/runs/best_ysfs_gat.pt` |
| 参考表 data1123 | `preprocess/data1123.csv` |

### B. 帕累托遗传逆设计（Pareto）

```bash
# 目标 YS/FS 为 data1123 物理量纲（默认）
python -m Pareto.run_ga_design --target-ys 1014.8 --target-fs 0.147

# 固定试验环境 tem/sr（可选，须成对提供）
python -m Pareto.run_ga_design --target-ys 1014.8 --target-fs 0.147 \
  --fixed-tem 25 --fixed-sr 0.001

# 拓展育种池 + 随机移民（默认关闭）
python -m Pareto.run_ga_design --target-ys 1014.8 --target-fs 0.147 \
  --breeder-pool expanded
```

输出目录 `Pareto/outputs_ga/`：`pareto_front.json`、`ga_summary.txt`、`pareto_scatter.png`；拓展育种时另有 `virtual_nodes_log.jsonl`。

详见 [Pareto/README.md](Pareto/README.md)。

### C. 梯度反推（grd）

```bash
python -m grd.run_inversion \
  --data-dir gnnDir/gnndataPT/r-gatPT \
  --ckpt modelAll/runs/best_rgat_full.pt \
  --rgat-dir modelAll \
  --out-dir grd/outputs
```

也可将 `--ckpt` / `--rgat-dir` 换为 `gnnDir/gnn/r-gatDouble` 路径。输出：`x_inv.pt`、`inversion_summary.json` / `.txt`。

**硬件**：全图 RGAT 反传建议 **NVIDIA GPU（≥16GB 显存）**；冒烟可加 `--force-cpu --inits training_mean`。

### D. 重新训练 GNN

**全量 604（推荐，与 Pareto 默认权重一致）：**

```bash
python modelAll/build_data.py --sanity
python modelAll/train.py
```

**gnnDir 原流程（r-gatDouble 等）：** 见 [gnnDir/README.md](gnnDir/README.md)、[modelAll/README.md](modelAll/README.md)。

### E. 从原始 CSV 生成 datagnn

```bash
# gnnDir 封装（写出 datagnn.csv + testenv_stats.csv）
python gnnDir/build_datagnn.py \
  --input preprocess/dataOri2.csv \
  --output gnnDir/datacsv/datagnn.csv

# 或 preprocess 模块直接 forward / verify
python -m preprocess.preprocess_datagnn_repro forward \
  --input preprocess/dataOri2.csv \
  --output gnnDir/datacsv/datagnn_repro.csv
```

### F. 符号蒸馏（symbolTorch，可选）

需 **Python ≥ 3.11** 单独环境，见 [symbolTorch/README.md](symbolTorch/README.md)。

---

## 特征与量纲（摘要）

- **30 维输入**：element(10) + testenv(2, z-score) + coldway(18)
- **Ti 余量**：`Ti = 100 − sum(10 元)`（A 模式）
- **Pareto 目标**：CLI 默认 **data1123** 量纲；内部转为 `ys.pt`/`fs.pt` 刻度优化；日志/JSON 逆变换回 data1123
- **FS 刻度**：`data1123` 的 `sr`/`FS`（如 `0.147`）≠ `dataOri2` 的 `fcr`（如 `14.7`）；勿混用

公式与常量见 [preprocess/PREPROCESS_datagnn.md](preprocess/PREPROCESS_datagnn.md)。

---

## 依赖

| 文件 | 适用范围 |
|------|----------|
| [`requirements.txt`](requirements.txt) | gnnDir + grd + Pareto + modelAll（Python 3.10） |
| [`symbolTorch/requirements.txt`](symbolTorch/requirements.txt) | 符号蒸馏（Python ≥ 3.11） |

核心：`torch`、`torch-geometric`、`numpy<2`、`pandas`。

---

## 常见问题

**Q：`ModuleNotFoundError: No module named 'grd'`**  
A：在仓库根目录执行，并 `export PYTHONPATH=$(pwd)`。

**Q：反推 CUDA OOM**  
A：更大显存 GPU，或减少 multistart（见 `grd/docs/run_inversion.md`）。

**Q：`build_datagnn.py` 或预处理报错**  
A：不再使用 `pt_dataset.py`。冷加工与 datagnn 复现见 `preprocess/preprocess_datagnn_repro.py`；构图见 `gnnDir/build_datagnn.py`。

**Q：Pareto 的 FS 目标填多少？**  
A：与 `preprocess/data1123.csv` 的 `FS` 列一致（如 `0.147`），不是 dataOri2 的 `20` 或 `14.7`。

**Q：symbolTorch 与主环境能否共用 Python 3.10？**  
A：不推荐；`torch-symbolic` 需 **Python ≥ 3.11**。

---

## 分支

| 分支 | 说明 |
|------|------|
| **`meta4TiiGnn`** | 当前主分支（gnnDir + preprocess + grd + Pareto + modelAll） |
| `master` | 与主开发同步时可用；新功能以 `meta4TiiGnn` 为准 |

```bash
git pull origin meta4TiiGnn
```

---

## 许可证与引用

子模块 README 含算法说明；反推文献见 [grd/README.md](grd/README.md)。
