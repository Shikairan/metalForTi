# scripts — 辅助脚本（初学者）

## check_env.py — 运行前自检

**作用**：在跑四个教例之前，检查 Python 版本、依赖、数据、教师权重是否齐全。

```bash
cd /home/data/metalTi/metalForTi/symbolTorch
/root/miniconda3/bin/python3.13 scripts/check_env.py
```

### 输出怎么读

| 显示 | 含义 | 怎么办 |
|------|------|--------|
| `[OK] symtorch` | SymTorch 已安装 | 继续 |
| `[FAIL] symtorch` | 未安装或 Python 太旧 | 用 3.11+ 执行 `pip install -r requirements.txt` |
| `Data dir: ... MISSING` | 没有图数据 | 到 `gnnDir` 跑 `regenerate_rgnnpt.py` |
| `Teacher ckpt: ... MISSING` | 没有教师 | 到 `r-gatDouble` 跑 `train_fs_gat.py` |
| `[WARN] Python >= 3.11` | 当前 Python 过低 | 换解释器 |

通过后再跑 `lowExp/run_distill.py --quick`。

完整教程见 [../README.md](../README.md)。
