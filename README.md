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

## 环境与硬件（必读）

### Python 版本

| 推荐 | 说明 |
|------|------|
| **Python 3.11** | **统一环境**：`gnnDir` 训练、`grd` 反推、`symbolTorch` 符号蒸馏均可使用 |

`torch-symbolic` 在 PyPI 上要求 **Python ≥ 3.11**，因此不再维护单独的 3.10 环境说明。

```bash
python --version   # 应显示 Python 3.11.x
```

### NVIDIA 驱动 / CUDA（参考：Driver 590 + CUDA 13.1）

`nvidia-smi` 示例：

```text
Driver Version: 590.48.01
CUDA Version: 13.1
```

说明：

- **「CUDA Version: 13.1」** 表示驱动支持的 **最高 CUDA 版本**，**不等于** 系统已安装 CUDA Toolkit。
- 安装 PyTorch 时请选 **`cu130`** 预编译包（自带 CUDA 运行时），与 **Driver 590** 兼容，**无需** 再装系统级 CUDA 13.1。
- 若 GPU 较老（如 Maxwell/Pascal/Volta），可改用 **`cu126`** 栈（见 [常见问题](#常见问题)）。

安装后自检：

```bash
nvidia-smi
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available(), torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

---

## 安装

### 1. 克隆与分支

```bash
git clone https://github.com/Shikairan/metalForTi.git
cd metalForTi
git checkout meta4TiiGnn
```

### 2. 创建虚拟环境（Python 3.11）

**conda（推荐）：**

```bash
conda create -n metal4ti python=3.11 -y
conda activate metal4ti
```

**venv：**

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. 安装 PyTorch（GPU / CUDA 13.x 驱动）

在仓库根目录执行。**必须先装 GPU 版 torch，再装其余依赖**（避免 pip 从 PyPI 装到 CPU 版覆盖）：

```bash
pip install --upgrade pip

# PyTorch 2.10 + CUDA 13.0 运行时（适配 Driver 590 / nvidia-smi CUDA 13.1）
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu130

# PyG 扩展算子（与 torch 2.10.0+cu130 匹配）
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.10.0+cu130.html

# 其余 Python 依赖（含 torch-geometric、symbolTorch）
pip install -r requirements.txt
```

**仅 CPU：**

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> 若需更新到 PyTorch 2.11，可将上述版本改为 `2.11.0` / `0.26.0` / `2.11.0`，索引仍为 `cu130`（见 [PyTorch 历史版本](https://pytorch.org/get-started/previous-versions/)）。

### 4. 设置 PYTHONPATH

`grd` 以包形式导入，需在 **仓库根目录** 运行，并保证根目录在 `PYTHONPATH` 中：

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

（可将该行写入 `~/.bashrc` 或在每次会话中执行。）

### 5. symbolTorch 补充说明

根目录 `requirements.txt` 已包含 `torch-symbolic`。首次运行 PySR 可能自动安装 **Julia**，耗时较长：

```bash
cd symbolTorch && python scripts/check_env.py
```

---

## 快速开始

以下命令均在 **仓库根目录**、已 `activate` 主环境（Python 3.11）且已设置 `PYTHONPATH` 的前提下执行。

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
| [`requirements.txt`](requirements.txt) | **统一环境**（Python 3.11；gnnDir + grd + symbolTorch） |
| [`gnnDir/requirements.txt`](gnnDir/requirements.txt) | 历史参考（旧版 torch 2.0.1 / cu118） |
| [`symbolTorch/requirements.txt`](symbolTorch/requirements.txt) | 子模块参考；主安装以根目录 `requirements.txt` 为准 |

核心依赖：`torch 2.10+cu130`（单独安装）、`torch-geometric 2.8`、`numpy`、`pandas`、`torch-symbolic`。

---

## 常见问题

**Q：`ModuleNotFoundError: No module named 'grd'`**  
A：在仓库根目录执行，并 `export PYTHONPATH=$(pwd)`。

**Q：反推 CUDA OOM**  
A：换更大显存 GPU，或减少 multistart 初始点（见 `grd/docs/run_inversion.md`）。

**Q：`build_datagnn.py` 报错缺少 `pt_dataset`**  
A：该脚本依赖仓库根目录的 `pt_dataset.py`（部分克隆可能未包含）；从完整仓库获取或将其加入 `PYTHONPATH`。

**Q：`torch.cuda.is_available()` 为 False**  
A：确认先安装了 **cu130** 索引的 torch，而非 PyPI 默认 CPU 包；驱动需 `nvidia-smi` 正常。

**Q：GPU 较老，cu130 无法运行**  
A：改用 **cu126** 栈，例如：
`pip install torch==2.10.0 ... --index-url https://download.pytorch.org/whl/cu126`，
PyG 轮子索引改为 `https://data.pyg.org/whl/torch-2.10.0+cu126.html`。

**Q：已有 checkpoint 用旧版 torch 2.0.1 训练，升级后能否加载？**  
A：一般可以 `load_state_dict`；若报错，在相同数据上短程微调或保留旧环境仅做推理对比。

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
