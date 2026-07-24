# symbolTorch 文档

当前仅保留 **lowExp**：把教师 GNN 的 YS/FS 预测蒸馏成**仅依赖 30 维表格特征**的可读符号公式（推理不再需要图）。

| 文档 | 内容 |
|------|------|
| [lowExp_overview.md](lowExp_overview.md) | 做什么、流水线、目录结构 |
| [lowExp_io.md](lowExp_io.md) | 接口：输入文件、CLI、输出产物与字段 |
| [lowExp_usage.md](lowExp_usage.md) | 环境、调用方式、常用示例 |
| [lowExp_qa.md](lowExp_qa.md) | 常见问题 |

入口脚本：[`../lowExp/run_distill.py`](../lowExp/run_distill.py)  
批量：[`../run_all.sh`](../run_all.sh)
