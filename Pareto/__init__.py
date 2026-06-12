"""
Pareto — 基因库累积式遗传逆设计（GNN 增广图 forward）。

主流程：
  1. 604 原始节点以物理实验标签 YS/FS 初始化基因库（最真实基准）。
  2. 每代通过加权锦标赛选父 → 分段交叉 → 三段式变异 → compile 约束修复
     → GNN 增广图 forward 评估子代适应度 → 累积入库。
  3. 帕累托前沿由 NSGA-II 非支配排序从精英池中提取，用于日志与最终报告。

CLI: python -m Pareto.run_ga_design --target-ys <float> --target-fs <float>
"""

__all__: list[str] = []
