# lowExp 常见问题（QA）

## Q1. lowExp 和以前的 highExp / medExp 有什么区别？

当前仓库**只保留 lowExp**。lowExp 在推理时**完全不用图**，只输出「30 维输入 → YS/FS」的公式。其它 Exp 目录已删除。

## Q2. 能不能自己选 GNN 的 `.pt`？

可以，用 **`--ckpt`**：

```bash
python run_distill.py --ckpt /path/to/best_xxx.pt --hidden-dim 64
```

默认是 `modelAll/ysFs/runs/best_rgat_full.pt`。  
`hidden_dim` 必须与 ckpt 里记录的一致，否则会报 `hidden_dim mismatch`。

## Q3. 可以用 `utsFs` / `utsFsAll` 的权重吗？

技术上若结构仍是 Dual 头且 `state_dict` 能 `strict` 加载，可以指定 `--ckpt`。  
但教师第一头是 **UTS** 不是 YS，导出的 `ys_tabular_sym.*` 实际是 **UTS 公式**。报告与下游解释时不要叫 YS。数据目录仍需带 `ys.pt` 标签文件（评估 MAE 相对该标签）；若标签仍是 YS 而教师是 UTS，`metrics` 会失去可比性。建议 YS 任务继续用 `ysFs`。

## Q4. 为什么公式比教师差？`graph_info_loss` 是什么？

教师用了邻居边信息；公式只用节点自己的 30 维。  
`graph_info_loss_val_mae_* = tabular_val_mae - teacher_val_mae`，表示去掉图信息后验证 MAE 变差多少（通常 ≥ 0）。

## Q5. `--quick` 和正式跑差在哪？

`--quick` 把 PySR `niterations` 降到 **40**（正式默认 400），公式更糙、更快，适合通路径。出报告请去掉 `--quick`。

## Q6. 报错找不到 checkpoint / 数据？

- ckpt：先确认路径；或按提示 `cd modelAll/ysFs && python build_data.py && python train.py`
- 数据：确认 `material_graph.pt`、`ys.pt`、`fs.pt`、masks 在 `--data-dir` 下；可用 `python scripts/check_env.py`

## Q7. `ModuleNotFoundError: symtorch` / Python 版本不够？

需要 **Python ≥ 3.11** 和 `pip install torch-symbolic`（见 `requirements.txt`）。不要用仅装了主工程依赖的 3.10 环境硬跑。

## Q8. CPU 上 `torch.compile` / dynamo 相关错误？

`run_distill.py` 已默认 `TORCHDYNAMO_DISABLE=1`。若仍有问题，确认该环境变量未被外部覆盖为启用。

## Q9. 公式里为什么是 `el_Si`、`fcr` 而不是 `Si`、`sr`？

- `Si` 与 SymPy 符号冲突，故特征名写成 `el_Si`
- 训练图 testenv 第二维在本数据管线里叫 `fcr`（与 `dataOri2` 对齐的命名）；即使 data1123 列名是 `sr`，**模型空间特征名仍是 `fcr`**

## Q10. `*_sym.pt` 加载失败？

导出依赖 pickle/`dill`。JSON 公式仍可用。可选安装：`pip install dill` 后重跑蒸馏。

## Q11. `slime: false` 是什么？

历史字段（旧 sampleExp）。lowExp 不使用 SLIME，恒为 `false`，可忽略。

## Q12. 结果写在哪里？会不会覆盖？

默认每次新建 `lowExp/runs/<时间戳…>/`，**不覆盖**旧结果；`runs/latest` 指向最近一次。  
可用 `--run-name foo` 指定子目录名，或 `--out-dir` 指定任意完整路径。`./run_all.sh` 也不会清空历史。

## Q13. MAE 是相对标签还是相对教师？

`metrics.json` 里 teacher / tabular 的 MAE 都是相对 **`ys.pt` / `fs.pt` 标签**。  
蒸馏拟合目标是教师预测；评估与训练脚本一致，用标签算 MAE。

## Q14. 文档里旧路径 `best_ysfs_gat.pt` / `metalTi` 还对吗？

不对。以本 `doc/` 与 `common/constants.py` 为准：默认 ckpt 为 `modelAll/ysFs/runs/best_rgat_full.pt`，仓库路径为 `/home/data/metalgnn/metalForTi`。

## Q15. 方程可读性差、只有一截 RHS？

新版本会导出：
- `equations.md`：完整 `YS = …` / `FS = …` + 变量表
- JSON 字段 `equation`（完整式），不再只有 `equations["0"]`

已有结果可看 `lowExp/runs/latest/equations.md` 或各次子目录。若公式本身项数很少，是 PySR 搜到的简化式（常见于 `--quick`）；要更「长」的式子请去掉 `--quick` 并提高 `--sr-niterations` 后重跑蒸馏。

## Q16. 为什么 YS 与 UTS 都会写 `tabular_fs`？会互相覆盖吗？

旧版默认写到共享的 `lowExp/SR_output/tabular_fs/`，两次跑会互相干扰。  
现已改为 **`runs/<本次运行>/SR_output/tabular_ys|uts/` 与 `…/tabular_fs/`**，按运行次数隔离。旧的 `lowExp/SR_output/` 可手动删除。
