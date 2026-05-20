# symbolTorch — RGAT_Dual 四档符号蒸馏

对 [`gnnDir/gnn/r-gatDouble`](../gnnDir/gnn/r-gatDouble) 已训练教师 `RGAT_Dual`（`SingleEncoder_DualRGAT`）做 [SymTorch](https://symtorch.readthedocs.io/) 符号蒸馏，得到可读公式并量化相对教师的误差。

## 四档实验

| 目录 | 策略 | 推理时是否用图 | 可解释性 |
|------|------|----------------|----------|
| [highExp](highExp/) | 仅 `ys_encoder` / `fs_encoder` 符号化 | 是（RGAT+Head 仍为神经网络） | 高：公式只含 30 维成分/环境/工艺 |
| [medExp](medExp/) | Encoder + `ys_head` / `fs_head` 符号化 | 是（仅 RGAT 为黑盒） | 中 |
| [sampleExp](sampleExp/) | SLIME 局部符号解释 | 是（扰动单节点特征后全图前向） | 样本级 |
| [lowExp](lowExp/) | 教师标签 → 纯表格 `x→YS/FS` | **否** | 最强（全局闭式，牺牲邻域信息） |

## 环境

- **Python ≥ 3.11**（`torch-symbolic` 要求；本机可用 `miniconda` 的 3.13）
- 与 `gnnDir` 的 PyTorch 2.0 / Python 3.10 **分开环境**更稳妥

```bash
cd metalForTi/symbolTorch
/root/miniconda3/bin/python3.13 -m pip install -r requirements.txt
/root/miniconda3/bin/python3.13 scripts/check_env.py
```

首次运行会拉取 Julia / PySR，可能需数分钟。

## 前置：图数据 + 教师权重

```bash
# 1) 生成 gnndataPT/r-gatPT（图、标签、mask）
cd metalForTi/gnnDir
python regenerate_rgnnpt.py --pt-bundle rgat

# 2) 训练双头 RGAT（若尚无 checkpoint）
cd gnn/r-gatDouble
python train_fs_gat.py \
  --data-dir ../../gnndataPT/r-gatPT \
  --out-dir runs \
  --epochs 1000
```

默认路径：

- 数据：`gnnDir/gnndataPT/r-gatPT/`
- 教师：`gnnDir/gnn/r-gatDouble/runs/best_ysfs_gat.pt`

## 一键运行

```bash
cd metalForTi/symbolTorch
chmod +x run_all.sh

./run_all.sh --quick    # 推荐首次：40 次 SR 迭代，encoder 仅 4 个输出维
./run_all.sh            # 完整蒸馏（encoder 64 维 × 2 分支，耗时长）
```

## 分档运行

```bash
PY=/root/miniconda3/bin/python3.13

$PY highExp/run_distill.py --quick
$PY medExp/run_distill.py --quick --encoder-sym-dir highExp/runs
$PY lowExp/run_distill.py --quick
$PY sampleExp/run_distill.py --quick
```

### 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--data-dir` | `gnnDir/gnndataPT/r-gatPT` | 图与标签 |
| `--ckpt` | `r-gatDouble/runs/best_ysfs_gat.pt` | 教师权重 |
| `--out-dir` | 各档 `runs/` | 输出目录 |
| `--quick` | 关 | 少迭代 + encoder 仅前 4 维 |
| `--sr-niterations` | 400 | PySR 迭代次数 |
| `--device` | auto | `cpu` / `cuda` |
| `--include-val` | 关 | 蒸馏样本是否包含验证集节点 |

## 输入特征命名（30 维）

与 `datagnn.csv` / `rgcn_dataloader` 列顺序一致，PySR 变量名为：

`Al, Zr, Sn, Mo, Cr, Nb, el_Si, V, Ta, Fe, tem, fcr, coldway_0 … coldway_17`

说明：`Si` 在 SymPy 中为保留名，故写作 `el_Si`。

## 输出说明

每档 `runs/` 下典型文件：

- `*_sym.json` — 符号公式（**建议纳入版本库**）
- `*_sym.pt` — 完整 SymbolicModel（体积大，已在 `.gitignore` 忽略）
- `metrics.json` — Teacher / Hybrid / Tabular 的 MAE
- `summary.md` — 简要文字摘要
- `sr_cache/` — PySR 中间结果（可删后重跑）

**断点续跑**：`highExp` / `medExp` 若中断，再次执行同一命令会尝试从 `runs/*_encoder_sym.pt` 续蒸馏未完成的输出维。

## 目录结构

```
symbolTorch/
  README.md
  requirements.txt
  run_all.sh
  scripts/check_env.py
  common/          # 数据加载、教师、蒸馏、混合模型
  highExp/
  medExp/
  lowExp/
  sampleExp/
```

## 引用

若使用 SymTorch，请引用其 [论文 / 文档](https://arxiv.org/abs/2602.21307)。
