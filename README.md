# metalForTi — 钛合金 YS/FS 图神经网络与梯度反推

基于材料组分、试验环境与工艺特征，用 **图神经网络（GNN）** 预测钛合金 **YS（屈服强度）** 与 **FS**，并支持在冻结模型上对 **30 维输入特征** 做 **梯度反推（inverse design）**。

**主开发分支：`meta4TiiGnn`**（本 README 对应该分支上的完整代码）。

---

## 仓库结构

| 目录 | 作用 | 详细文档 |
|------|------|----------|
| [`gnnDir/`](gnnDir/) | 数据构建、异质图 PT、RGAT/R-GCN **训练** | [gnnDir/README.md](gnnDir/README.md) |
| [`grd/`](grd/) | 冻结 GNN 下的 **全特征梯度反推** | [grd/README.md](grd/README.md) |
| [`symbolTorch/`](symbolTorch/) | 将 GNN 蒸馏为 **可读符号公式**（SymTorch + PySR） | [symbolTorch/README.md](symbolTorch/README.md) |
| [`optimizeSympy/`](optimizeSympy/) | SymPy 相关优化（占位/扩展） | — |

典型流水线：

```text
原始 CSV → gnnDir 构图/训练 → best_ysfs_gat.pt
                ↓
         grd 梯度反推（目标 YS/FS → 反求特征）
                ↓
      symbolTorch 符号蒸馏（可选，解释性公式）
```

---

## Python 版本（必读）

本仓库 **不同子模块对 Python 版本要求不同**，请按用途选择环境：

| 用途 | 推荐 Python | 说明 |
|------|-------------|------|
| **gnnDir 训练 + grd 反推**（主路径） | **3.10** | 与 `requirements.txt` 中 `torch==2.0.1` 测试栈一致；**首选** |
| **symbolTorch 符号蒸馏** | **≥ 3.11** | `torch-symbolic` 要求；建议单独虚拟环境 |
| 不推荐 | 3.9 及以下 / 未验证的 3.13+ | 未在仓库内系统测试 |

**结论：**

- 只做 **训练 + 反推** → 使用 **Python 3.10**
- 还要跑 **symbolTorch** → 再建一个 **Python 3.11+** 环境，或在本机用 conda 管理两个 env

检查当前版本：

```bash
python --version   # 应显示 Python 3.10.x（主环境）
```

---

## 安装

### 1. 克隆与分支

```bash
git clone https://github.com/Shikairan/metalForTi.git
cd metalForTi
git checkout meta4TiiGnn
```

### 2. 创建虚拟环境（Python 3.10）

**conda（推荐）：**

```bash
conda create -n metal4ti python=3.10 -y
conda activate metal4ti
```

**venv：**

```bash
python3.10 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. 安装 PyTorch

在仓库根目录执行。**GPU（CUDA 11.8 示例）请先装 torch，再装其余依赖：**

```bash
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**仅 CPU：**

```bash
pip install torch==2.0.1
pip install -r requirements.txt
```

### 4. 设置 PYTHONPATH

`grd` 以包形式导入，需在 **仓库根目录** 运行，并保证根目录在 `PYTHONPATH` 中：

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

（可将该行写入 `~/.bashrc` 或在每次会话中执行。）

### 5. 可选：symbolTorch 环境（Python ≥ 3.11）

```bash
conda create -n metal4ti-sym python=3.11 -y
conda activate metal4ti-sym
pip install torch>=2.0.0
pip install -r symbolTorch/requirements.txt
```

首次运行 PySR 可能自动安装 **Julia**，耗时较长。

---

## 快速开始

以下命令均在 **仓库根目录**、已 `activate` 主环境（Python 3.10）且已设置 `PYTHONPATH` 的前提下执行。

### A. 检查数据与权重（默认已随仓库提供）

| 资源 | 默认路径 |
|------|----------|
| 图数据 PT | `gnnDir/gnndataPT/r-gatPT/` |
| 最佳权重 | `gnnDir/gnn/r-gatDouble/runs/best_ysfs_gat.pt` |

### B. 梯度反推（grd）

```bash
python -m grd.run_inversion \
  --data-dir gnnDir/gnndataPT/r-gatPT \
  --ckpt gnnDir/gnn/r-gatDouble/runs/best_ysfs_gat.pt \
  --rgat-dir gnnDir/gnn/r-gatDouble \
  --out-dir grd/outputs
```

输出：

- `grd/outputs/x_inv.pt` — 反推特征张量
- `grd/outputs/inversion_summary.json` — 指标（机器可读）
- `grd/outputs/inversion_summary.txt` — 中文汇总报告

**硬件建议：** 全图 RGAT 反传 **强烈建议 NVIDIA GPU（≥16GB 显存）**；CPU 易 OOM，可加 `--force-cpu --inits training_mean` 做冒烟测试。

### C. 重新训练 GNN（gnnDir）

```bash
cd gnnDir/gnn/r-gatDouble
python train_fs_gat.py --data-dir ../../gnndataPT/r-gatPT --device cuda
```

数据再生与划分见 [gnnDir/README.md](gnnDir/README.md)。

### D. 符号蒸馏（symbolTorch，可选）

```bash
cd symbolTorch
python scripts/check_env.py    # 环境自检
# 按 symbolTorch/README.md 选择 lowExp / medExp / highExp 等
```

---

## 特征与模型约定（摘要）

- **输入维度**：30 = element(10) + testenv(2, z-score) + coldway(18)
- **元素列**：`Al, Zr, Sn, Mo, Cr, Nb, Si, V, Ta, Fe`（**Ti 为余量**：`Ti = 100 − sum(element)`，A 模式）
- **正向模型**：`SingleEncoder_DualRGAT`（双头 RGAT，共享编码）
- **反推目标**：使预测 YS/FS 逼近给定目标（默认 ground truth）

详见 [grd/README.md](grd/README.md#特征布局30-维) 与 [grd/docs/](grd/docs/)。

---

## 依赖说明

| 文件 | 适用范围 |
|------|----------|
| [`requirements.txt`](requirements.txt) | **gnnDir + grd**（Python 3.10，主环境） |
| [`gnnDir/requirements.txt`](gnnDir/requirements.txt) | 与根目录 requirements 等价参考 |
| [`symbolTorch/requirements.txt`](symbolTorch/requirements.txt) | 符号蒸馏（Python ≥ 3.11） |

核心依赖：`torch`、`torch-geometric`、`numpy<2`、`pandas`。

---

## 常见问题

**Q：`ModuleNotFoundError: No module named 'grd'`**  
A：在仓库根目录执行，并 `export PYTHONPATH=$(pwd)`。

**Q：反推 CUDA OOM**  
A：换更大显存 GPU，或减少 multistart 初始点（见 `grd/docs/run_inversion.md`）。

**Q：`build_datagnn.py` 报错缺少 `pt_dataset`**  
A：该脚本依赖仓库根目录的 `pt_dataset.py`（部分克隆可能未包含）；从完整仓库获取或将其加入 `PYTHONPATH`。

**Q：symbolTorch 与 gnnDir 能否共用一个 Python 3.10 环境？**  
A：不推荐。`torch-symbolic` 需要 **Python ≥ 3.11**，请单独建环境。

---

## 许可证与引用

子模块 README 中含算法说明与参考文献；反推算法详见 [grd/README.md#参考文献与链接](grd/README.md#参考文献与链接)。

---

## 分支说明

| 分支 | 状态 |
|------|------|
| **`meta4TiiGnn`** | **当前主分支**，含完整 `gnnDir` + `grd` + 文档 |
| `master` | 与 `meta4TiiGnn` 同步（内容相同）；新开发请以 `meta4TiiGnn` 为准 |

```bash
git pull origin meta4TiiGnn
```
