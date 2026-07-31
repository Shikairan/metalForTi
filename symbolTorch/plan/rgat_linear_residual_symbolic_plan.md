# RGAT 线性基线与非线性残差符号蒸馏开发计划

## 1. 计划目标

本计划用于把现有 `symbolTorch/lowExp` 扩展为一条职责清晰、可单独验证、可联合运行的 RGAT 符号蒸馏流程。最终模型由两部分组成：

```text
RGAT 教师输出 = 30 维线性基线 + 非线性符号残差
```

对每个预测头分别构造：

```text
Y_head0(x) = L_head0(x) + R_head0(x)
Y_FS(x)    = L_FS(x)    + R_FS(x)
```

其中：

- `x` 是 `datagnnUts.csv` 中顺序固定的 30 维模型空间输入。
- `L(x)` 由新增的 `liner` 模块拟合，目标是蒸馏 RGAT 教师输出，而不是直接拟合 CSV 中的真实标签。
- `R(x)` 由现有 `lowExp` 生成，拟合目标改为 `RGAT 教师输出 - 线性基线输出`。
- `comb` 负责统一调用、结果组合、指标评估、模型空间公式输出和人类可读物理空间公式解释。
- 第一预测头必须显式选择 `YS` 或 `UTS`，第二预测头为 `FS`。

## 2. 强制目录约束

开发时必须形成以下结构，目录名称必须严格使用 `liner`、`lowExp`、`comb`，不得把三者合并成单个脚本：

```text
symbolTorch/
├── liner/
│   ├── run_liner.py
│   └── runs/
├── lowExp/
│   ├── run_distill.py
│   └── runs/
├── comb/
│   ├── run_comb.py
│   └── runs/
├── common/
└── plan/
    └── rgat_linear_residual_symbolic_plan.md
```

强约束如下：

1. `liner` 只负责 RGAT 教师预测的线性蒸馏、线性公式导出和残差生成。
2. `lowExp` 只负责非线性残差的符号回归，不再直接拟合完整 RGAT 输出。
3. `comb` 是面向用户的唯一联合入口，负责调用 `liner` 和 `lowExp`，但不得在内部复制两者的拟合实现。
4. `comb` 必须生成最终完整公式、模型空间说明、人类可读物理空间解释和联合评估指标。
5. 公共的数据校验、教师前向、公式记录、编解码和指标逻辑放在 `common` 中，避免三个目录各自维护不同实现。
6. 本计划阶段只创建本 Markdown，不创建或修改任何 Python 开发代码。

## 3. 数据与蒸馏定义

### 3.1 输入数据

线性拟合的输入来源固定为：

```text
/home/data/metalgnn/metalForTi/gnnDir/datacsv/datagnnUts.csv
```

只读取前 30 列作为输入，并严格保持 CSV 列顺序：

```text
element_0 ... element_9,
testenv_0, testenv_1,
coldway_0 ... coldway_17
```

公式内部变量映射保持现有 `symbolTorch` 约定：

```text
element_0..9  -> Al, Zr, Sn, Mo, Cr, Nb, el_Si, V, Ta, Fe
testenv_0..1  -> tem, fcr
coldway_0..17 -> coldway_0..17
```

`Si` 在可解析公式中继续使用 `el_Si`，避免与 SymPy 名称冲突。

### 3.2 蒸馏目标

`datagnnUts.csv` 中的 `YS`、`FS`、`UTS` 是真实标签，不作为线性蒸馏的直接训练目标。正确目标由指定 checkpoint 的 RGAT 教师在完整图上产生：

```text
teacher_head0, teacher_fs = RGAT(x_graph, edge_index, edge_type)
```

随后使用同一批样本、同一行顺序的 CSV 30 维输入进行线性拟合：

```text
linear_head0 = fit(x_csv_train, teacher_head0_train)
linear_fs    = fit(x_csv_train, teacher_fs_train)
```

残差定义固定为：

```text
residual_head0 = teacher_head0 - linear_head0
residual_fs    = teacher_fs    - linear_fs
```

`lowExp` 只拟合上述残差：

```text
symbolic_residual_head0 = PySR(x_train, residual_head0_train)
symbolic_residual_fs    = PySR(x_train, residual_fs_train)
```

### 3.3 样本对齐约束

CSV 行与图节点必须一一对应。开发时必须增加以下阻断式校验：

- CSV 数据行数等于图中 `sample` 节点数。
- CSV 前 30 列数值与 `graph["sample"].x` 在允许浮点误差内一致。
- `train_mask`、`val_mask` 长度与 CSV 行数一致。
- 线性预测、残差、教师预测均保存样本索引，组合阶段按索引核对，不允许只依赖文件行位置静默拼接。
- 任一校验失败时立即退出，不生成公式。

### 3.4 训练与验证边界

- 默认仅使用 `train_mask` 拟合线性模型和非线性残差公式。
- `val_mask` 只用于评估，不得参与系数估计、PySR 搜索或公式选择。
- `train_mask` 与 `val_mask` 必须非空且互斥；若数据采用全量训练模式，指标不得继续命名为独立验证指标。
- 线性方法、Ridge `alpha` 和 PySR 候选选择如需调参，只能在训练集内部交叉验证；外层 `val_mask` 只在最终公式固定后评估一次。
- 完全相同的 30 维输入应按输入指纹分组划分，避免同一输入的重复行跨越训练集和验证集。
- 若保留 `--include-val` 研究选项，输出必须显式标记结果不再是独立验证结果。
- 教师前向可在完整图上执行，因为 RGAT 本身需要图邻居；公式拟合样本仍由 mask 控制。

## 4. 三部分代码级实现方案

本节是实现时必须遵循的主流程。线性、非线性和组合三个阶段必须通过明确的数组与产物契约衔接，不允许通过临时字符串或隐式全局状态传递结果。

### 4.1 统一张量与数组契约

三个阶段统一使用以下形状和数据类型：

```text
X_all               float64 [N, 30]
train_mask          bool    [N]
val_mask            bool    [N]
sample_id           int64   [N]
teacher_head0       float64 [N]
teacher_fs          float64 [N]
linear_head0        float64 [N]
linear_fs           float64 [N]
residual_head0      float64 [N]
residual_fs         float64 [N]
symbolic_residual_* float64 [N]
combined_*          float64 [N]
```

RGAT 可以在原训练 dtype 和设备上前向，但进入线性拟合、残差计算、公式回放和指标计算前统一转为 CPU `float64`。所有阶段都必须检查 shape、样本数量、样本 ID 和 `isfinite`，禁止依赖 NumPy 广播把 `[N]` 与 `[N,1]` 静默相减。

### 4.2 线性部分的具体构建方案

默认实现使用带截距、截距不参与惩罚的 Ridge。当前 30 维数据存在常量列和共线性，普通 OLS 的 30 个系数不唯一，因此 Ridge 应作为默认路径；OLS 只作为诊断和对照路径。

为保持与 RGAT 完全一致的输入语义，线性模型直接使用 `datagnnUts.csv` 前 30 列的模型空间数值，不再进行第二次特征标准化。Ridge 的 `alpha` 只允许通过训练集内部交叉验证选择，并在 manifest 中记录候选集合、折分和最终值。

线性求解步骤：

```python
X_train = X_all[train_mask]
y_train = teacher_target[train_mask]

X_aug = concatenate([ones((len(X_train), 1)), X_train], axis=1)
ridge_rows = sqrt(alpha) * eye(31)
ridge_rows[0, 0] = 0.0

A = concatenate([X_aug, ridge_rows], axis=0)
b = concatenate([y_train, zeros(31)], axis=0)
beta = lstsq(A, b)
intercept = beta[0]
coefficients = beta[1:]
linear_all = intercept + X_all @ coefficients
residual_all = teacher_target - linear_all
```

实际代码不得直接对病态矩阵求逆。建议函数边界：

```python
def fit_linear_teacher(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    method: str,
    alpha: float,
) -> LinearFitResult:
    ...

def predict_linear(
    x: np.ndarray,
    intercept: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    ...

def compute_teacher_residual(
    teacher: np.ndarray,
    linear: np.ndarray,
) -> np.ndarray:
    ...
```

求解器规则：

- Ridge 要求 `alpha > 0`，使用增广最小二乘的 QR/SVD 求解，截距不惩罚；不得构造 `X.T @ X` 放大条件数。
- OLS 使用 `numpy.linalg.lstsq` 的 SVD 最小范数解，并固定、记录 `rcond`；不得使用显式矩阵求逆。
- 拟合前记录增广设计矩阵的秩、奇异值、条件数和常量列。
- 若 `rank < 31`，必须记录 `rank_deficient=true` 和 `coefficient_identifiability=false`。
- 无论系数是否为零，JSON 和公式都按固定顺序输出全部 30 项。
- 线性公式中的完整性是结构要求，不代表每个系数可辨识或具有显著物理贡献。

两个预测头分别调用同一个实现：

```python
fit_head0 = fit_linear_teacher(X_train, teacher_head0[train_mask], method=method, alpha=alpha)
fit_fs = fit_linear_teacher(X_train, teacher_fs[train_mask], method=method, alpha=alpha)

linear_head0 = predict_linear(X_all, fit_head0.intercept, fit_head0.coefficients)
linear_fs = predict_linear(X_all, fit_fs.intercept, fit_fs.coefficients)

residual_head0 = compute_teacher_residual(teacher_head0, linear_head0)
residual_fs = compute_teacher_residual(teacher_fs, linear_fs)
```

`residual_targets.pt` 必须保存 `X_all`、两个教师输出、两个线性输出、两个残差、mask、sample ID、特征名和 manifest 指纹。这样 `lowExp` 不需要重新运行 RGAT 或重新拟合线性模型。

### 4.3 剩余非线性部分的 SymbolTorch 构建方案

非线性目标不是原始标签，也不是完整 RGAT 输出，而是线性模型未解释的逐样本教师残差：

```text
r_head0[i] = teacher_head0[i] - linear_head0[i]
r_fs[i]    = teacher_fs[i]    - linear_fs[i]
```

SymbolTorch 1.0.1 的 `SymbolicModel.distill` 没有 `distill(X, y)` 形式的显式目标参数。它的实际行为是先调用构造时传入的 `block(X)` 收集目标，再在内部执行 PySR 的 `fit(X, y)`。因此残差拟合必须增加一个严格绑定目标的 callable，不能继续使用现有的最近邻 `make_tabular_lookup_fn`。

现有最近邻 lookup 必须退出残差主路径，原因是完全重复的输入行会永远命中第一个目标，导致后续重复样本残差被静默丢弃；近似输入还可能映射到错误样本。

计划新增公共帮助函数，职责如下：

```python
def make_bound_target_fn(
    x_expected: np.ndarray,
    y_expected: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    x_ref = as_float32_2d(x_expected)
    y_ref = as_float32_column(y_expected)

    def target_fn(x_received: np.ndarray) -> np.ndarray:
        x = as_float32_2d(x_received)
        if x.shape != x_ref.shape:
            raise ValueError("unexpected SymbolTorch input shape")
        if not np.array_equal(x, x_ref):
            raise ValueError("SymbolTorch changed input values or row order")
        return y_ref.copy()

    return target_fn
```

该 callable 是“本次训练矩阵到本次监督残差”的严格绑定，不进行最近邻搜索、不按特征去重，并保留相同 `X` 对应的每一条残差观测。

残差蒸馏函数必须直接围绕 SymbolTorch 构建：

```python
def distill_residual_targets(
    x_train: np.ndarray,
    residual_train: np.ndarray,
    *,
    block_name: str,
    variable_names: list[str],
    sr_params: dict,
    save_path: Path,
):
    target_fn = make_bound_target_fn(x_train, residual_train)
    sym = SymbolicModel(target_fn, block_name=block_name)
    sym.distill(
        x_train.astype(np.float32),
        sr_params=sr_params,
        fit_params={"variable_names": variable_names},
        save_path=str(save_path),
        SLIME=False,
    )
    sym.switch_to_symbolic()
    return sym
```

这里 `SLIME=False` 是强制要求。SLIME 会生成新的合成输入并再次调用 block，但预先计算的 RGAT 残差只定义在原图节点样本上，合成输入没有对应的图邻域和教师残差。

双头调用过程固定为：

```python
payload = load_and_validate_liner_run(args.liner_run)
X_train = payload.X_all[payload.train_mask]

sym_residual_head0 = distill_residual_targets(
    X_train,
    payload.residual_head0[payload.train_mask],
    block_name=f"residual_{payload.head0_name.lower()}",
    variable_names=FEATURE_NAMES,
    sr_params=sr_params,
    save_path=sr_output_root,
)

sym_residual_fs = distill_residual_targets(
    X_train,
    payload.residual_fs[payload.train_mask],
    block_name="residual_fs",
    variable_names=FEATURE_NAMES,
    sr_params=sr_params,
    save_path=sr_output_root,
)
```

生成符号模型后，在 train、val 和全体样本上分别执行：

```python
rhat_head0 = evaluate_symbolic(sym_residual_head0, X_all)
rhat_fs = evaluate_symbolic(sym_residual_fs, X_all)
```

必须验证输出 shape 为 `[N]` 且全部有限。PySR 允许 `/`、`inv`、`log`、`exp`，所以候选公式若在 train、val 或全体样本上产生 `NaN/Inf`，本次运行判定失败，不得写成功标记。

重复输入不得删除。实现时按完全一致的 30 维输入分组，报告组内教师输出和残差方差；若同一输入存在不同残差，确定性无图公式无法同时精确拟合，必须将组内误差下界记录为不可约蒸馏误差，而不是归因于 SymbolTorch 搜索失败。

### 4.4 线性与非线性的结合方案

组合阶段不重新训练任何模型，只读取经过 manifest 指纹校验的 `liner` 和 `lowExp` 产物：

```python
linear = load_liner_result(liner_run)
residual = load_lowexp_result(lowexp_run)
assert_compatible(linear.manifest, residual.manifest)

combined_head0 = linear.linear_head0 + residual.symbolic_residual_head0
combined_fs = linear.linear_fs + residual.symbolic_residual_fs
```

最终公式在数学上是：

```text
F_head0(x) = intercept_head0
             + sum(coef_head0[j] * x[j], j=0..29)
             + R_head0(x)

F_FS(x) = intercept_fs
          + sum(coef_fs[j] * x[j], j=0..29)
          + R_FS(x)
```

公式不得仅通过字符串拼接后直接 `eval`。组合阶段应从 PySR regressor 的 `sympy()` 结果取得受控 SymPy 表达式，在表达式树层构建加法：

```python
linear_expr = build_linear_sympy_expr(intercept, coefficients, FEATURE_NAMES)
residual_expr = selected_pysr_regressor.sympy()
validate_expression_tree(residual_expr, allowed_symbols, allowed_operators)
combined_expr = sympy.Add(linear_expr, residual_expr, evaluate=False)
```

计划新增统一表达式接口：

```python
def build_linear_sympy_expr(...) -> sympy.Expr: ...
def validate_expression_tree(...) -> None: ...
def serialize_expression_ir(...) -> dict: ...
def evaluate_expression_numpy(...) -> np.ndarray: ...
```

白名单只允许 30 个特征符号、数值常量和配置中启用的 `+ - * / sin exp log inv`，禁止未知符号、属性访问、下标和任意函数调用。JSON 同时保存结构化表达式 IR、原始 SymPy 字符串、展示字符串和 evaluator 版本。

组合结果必须通过两条独立数值路径交叉校验：

```python
combined_from_modules = linear_prediction + symbolic_residual_prediction
combined_from_formula = evaluate_expression_numpy(combined_expr, X_all)

assert_allclose(combined_from_formula, combined_from_modules, rtol=..., atol=...)
```

只有 train、val、全体样本均满足 shape、有限值和数值一致性，才允许生成最终成功产物并更新 `latest`。

### 4.5 代码调用顺序

端到端调用顺序必须清晰固定：

```mermaid
flowchart LR
    A[加载 CSV、图、mask、checkpoint] --> B[RGAT 完整图教师前向]
    B --> C[liner 仅用训练集拟合 Ridge]
    C --> D[全体样本计算 teacher-linear 残差]
    D --> E[lowExp 绑定 X_train 与 residual_train]
    E --> F[SymbolTorch 内部调用 PySR fit]
    F --> G[全体样本回放残差公式]
    G --> H[comb 在表达式树层相加]
    H --> I[数值一致性与有限值验证]
    I --> J[模型空间公式与物理空间解释]
```

失败传播规则：任一阶段失败，`comb` 立即失败；不得继续使用半成品，不得生成 `_SUCCESS`，不得把 `latest` 指向失败目录。

## 5. `liner` 模块计划

### 5.1 职责

`liner` 是独立的线性 RGAT 蒸馏模块，负责：

1. 加载 `datagnnUts.csv` 的 30 维模型空间输入。
2. 加载图、mask、RGAT checkpoint 并得到两个教师预测头。
3. 对两个预测头分别拟合线性模型。
4. 计算全体样本的线性预测和教师残差。
5. 导出完整 30 项线性公式、系数、截距、预测、残差和指标。

### 5.2 线性模型约束

- 支持普通最小二乘和 Ridge，默认使用 `alpha > 0` 的 Ridge；方法与 alpha 只能预先指定或通过训练集内部交叉验证确定，不得使用外层验证集选择。
- 若使用 Ridge，`alpha` 必须是显式参数并写入产物清单。
- 30 个变量必须全部保存在系数表和最终线性公式中。
- 公式格式化时不得因系数较小而删除变量项，也不得自动把近零系数显示成零。
- 保存系数时使用足够精度；展示公式可另行控制有效位数，但机器可读公式不得使用低精度展示值替代原始系数。
- “30 个变量均出现”只代表公式结构完整，不得解释为 30 个变量均具有显著物理贡献。

线性公式标准形式：

```text
L_target(x) = intercept + coef_0*x_0 + ... + coef_29*x_29
```

### 5.3 `liner` 输入接口

计划中的最小参数集合：

```text
--csv
--data-dir
--ckpt
--head0-name {YS,UTS}
--head1-name FS
--method {ols,ridge}
--alpha
--device
--seed
--out-dir 或 --run-name
```

### 5.4 `liner` 输出契约

每次运行至少生成：

```text
liner/runs/<run_id>/
├── manifest.json
├── teacher_predictions.pt
├── linear_head0.json
├── linear_fs.json
├── linear_predictions.pt
├── residual_targets.pt
├── linear_equations.md
└── metrics.json
```

关键字段：

- `manifest.json`：输入绝对路径、checkpoint、目标头名称、样本数、特征顺序、mask 摘要、方法、alpha、seed 和运行标识。
- `linear_*.json`：原始高精度截距、按 30 维顺序保存的系数、变量名、完整机器可读 RHS 和展示 RHS。
- `residual_targets.pt`：两个头的训练/验证/全体残差、样本索引、特征矩阵摘要及来源 manifest 标识。
- `metrics.json`：教师对真实标签、线性模型对教师、线性模型对真实标签的 train/val MAE，并可补充 RMSE、R²。

## 6. `lowExp` 非线性残差计划

### 6.1 职责调整

保留 `lowExp` 作为 PySR/SymTorch 非线性模块，但将它从“直接拟合教师完整输出”调整为“拟合 `liner` 产生的教师残差”。

新的核心关系：

```text
R_target(x) ≈ teacher_target(x, graph) - L_target(x)
```

### 6.2 输入方式

`lowExp` 必须接收 `liner` 的运行目录或明确的 `residual_targets.pt` 与 `manifest.json`，不得自行重新拟合线性模型。建议接口：

```text
--liner-run <path>
--sr-niterations
--sr-maxsize
--quick
--device
--seed
--out-dir 或 --run-name
```

`lowExp` 启动时必须校验：

- 输入残差存在且包含两个目标头。
- 30 维特征名、顺序、样本索引与 `liner` manifest 一致。
- 目标头名称与 checkpoint 任务一致。
- 训练和验证 mask 未被改变。
- 非线性公式拟合使用训练残差，验证残差只用于评价。

### 6.3 输出契约

每次运行至少生成：

```text
lowExp/runs/<run_id>/
├── manifest.json
├── head0_residual_sym.json
├── fs_residual_sym.json
├── head0_residual_sym.pt
├── fs_residual_sym.pt
├── residual_equations.md
├── residual_predictions.pt
├── metrics.json
└── SR_output/
```

指标至少包括：

- 零残差基线 MAE。
- 符号残差对教师残差的 train/val MAE。
- 每条残差公式的复杂度和实际使用变量。
- train/val 的 `R² = 1 - SSE_linear / SST_teacher`、残差标准差和 `Cov(linear, residual)`；不得把单独的残差方差比直接称为线性解释比例。

## 7. `comb` 联合模块计划

### 7.1 职责

`comb` 是最终面向用户的统一入口，负责：

1. 接收数据、checkpoint、目标头和线性/PySR 参数。
2. 创建统一运行 ID 和联合输出目录。
3. 调用 `liner` 生成线性模型及残差。
4. 把本次 `liner` 产物传给 `lowExp` 生成非线性残差公式。
5. 校验两个阶段的 manifest、样本索引、目标名称和特征顺序。
6. 组合机器可读完整公式。
7. 对最终公式执行数值回放，验证“组合公式输出”等于“线性输出 + 残差公式输出”。
8. 生成模型空间公式、人类可读物理空间公式说明和最终指标。

### 7.2 最终公式

模型空间最终公式固定为：

```text
head0_model = L_head0(x_model) + R_head0(x_model)
FS_model    = L_FS(x_model)    + R_FS(x_model)
```

组合时必须保留三个层次，不能只保留拼接后的字符串：

- `linear_equation_raw`
- `residual_equation_raw`
- `combined_equation_raw`

这样既能解释各部分贡献，也能对组合公式进行机器校验。

### 7.3 联合入口建议

计划中的统一调用形式：

```text
python comb/run_comb.py \
  --csv /home/data/metalgnn/metalForTi/gnnDir/datacsv/datagnnUts.csv \
  --data-dir <图数据目录> \
  --ckpt <RGAT checkpoint> \
  --head0-name UTS \
  --head1-name FS \
  --linear-method ridge \
  --alpha <值> \
  --sr-niterations <值> \
  --sr-maxsize <值> \
  --device <cpu或cuda>
```

`comb` 应支持复用已完成阶段：

- `--liner-run`：跳过线性阶段，复用经过校验的 `liner` 结果。
- `--lowexp-run`：跳过符号搜索，复用经过校验的 `lowExp` 结果。
- 复用时必须检查 manifest 指纹，不允许组合来自不同数据、checkpoint、目标或特征顺序的产物。

### 7.4 `comb` 输出契约

每次联合运行至少生成：

```text
comb/runs/<run_id>/
├── manifest.json
├── final_equations.json
├── final_equations_model_space.md
├── final_equations_physical.md
├── predictions.pt
├── metrics.json
├── summary.md
├── liner_run.txt
└── lowexp_run.txt
```

`final_equations.json` 对每个目标至少记录：

```text
target
feature_names
linear_intercept
linear_coefficients
linear_equation_raw
residual_equation_raw
combined_equation_raw
model_to_physical_input_transform
model_to_physical_output_transform
units
```

## 8. 编解码与人类可读公式

### 8.1 唯一编解码依据

模型空间与人类物理空间之间的转换必须复用或严格对齐：

```text
/home/data/metalgnn/metalForTi/preprocess/preprocess_datagnn_repro.py
```

不得在 `liner`、`lowExp`、`comb` 中分别硬编码三套不同统计量。开发时应把需要的纯转换能力作为公共接口复用，并把实际使用的均值、标准差、标签均值和 `T_DIV` 写入最终 manifest。

### 8.2 输入空间解码

元素 10 维保持 wt% 原值。试验环境需要将物理量编码成模型输入：

```text
testenv_0 = (tem - mean_tem) / std_tem
testenv_1 = (fcr_or_sr - mean_fcr) / std_fcr
```

最终解释文档中必须同时说明：

- 模型内部第二维名称为 `fcr`。
- 对 `data1123` 的人类输入，它与 `sr` 使用同一数值管线。
- 不能只把公式中的 `fcr` 文本替换为 `sr`，还必须展示 Z-score 代换。

Coldway 18 维按 3 个阶段、每阶段 3 种冷却方式、每种方式一对温度/时间展开：

```text
coldway_(6*i+2*j)   = active(i,j) * ln(T_i / 800)
coldway_(6*i+2*j+1) = active(i,j) * ln(t_i)
```

其中 `i=0..2`，`j=0..2`；未激活或无效阶段代入零。解释文档必须说明 `[0, 0]` 同时可能表示“未激活”和物理值 `T=800、t=1`，仅靠 18 维编码无法无歧义逆推激活状态。

### 8.3 输出空间解码

模型输出转换到物理量时使用预处理脚本中的标签均值：

```text
YS_MPa  = YS_model  * mean_YS
UTS_MPa = UTS_model * mean_UTS
FS_dataOri2 = FS_model * mean_FS
FS_data1123 = FS_dataOri2 / 100
```

最终文档必须明确 FS 展示采用 `dataOri2` 刻度还是 `data1123` 刻度，推荐同时给出两种换算，避免把相差 100 倍的结果混用。

### 8.4 两类最终文档

`comb` 必须同时生成：

1. `final_equations_model_space.md`：准确展示可直接作用于 CSV 前 30 列的公式，适合复现和数值验证。
2. `final_equations_physical.md`：展示物理字段如何经过编码进入公式、模型输出如何解码，并解释每个变量、单位、工艺槽位和不可逆边界。

物理空间文档不得声称已把带条件的 coldway 编码完全代数消除；应采用“完整模型公式 + 明确变量代换规则”的形式，保证准确性和可读性。

## 9. 指标与验收标准

### 9.1 分阶段指标

对两个预测头分别计算：

- RGAT 教师对真实标签的 train/val MAE。
- 线性模型对 RGAT 教师预测的 train/val MAE。
- 非线性符号模型对教师残差的 train/val MAE。
- 最终组合公式对 RGAT 教师预测的 train/val MAE。
- 最终组合公式对真实标签的 train/val MAE。
- `distillation_mae = MAE(combined, teacher)`，表示无图公式对教师的蒸馏误差。
- `teacher_label_mae = MAE(teacher, label)` 与 `student_label_mae = MAE(combined, label)`。
- `label_mae_delta = student_label_mae - teacher_label_mae`；该差值不得直接命名为纯图信息损失。

建议补充 RMSE 和 R²，但 MAE 必须保留为主指标，以便与现有 `lowExp` 对比。

### 9.2 数值一致性验收

必须通过以下自动化检查：

1. 线性 JSON 重新计算的输出与 `liner` 保存预测一致。
2. 残差定义满足 `teacher - linear = residual`，误差仅允许来自浮点精度。
3. `lowExp` 输出公式重新计算结果与保存的残差预测一致。
4. 组合结果满足 `combined = linear + symbolic_residual`。
5. `final_equations.json` 的机器可读公式回放结果与组合模型输出一致。
6. 模型空间输入经过解码再编码后，在非歧义样本上恢复到允许误差范围。
7. YS、UTS、FS 的输出解码与 `preprocess_datagnn_repro.py` 一致。

### 9.3 功能验收

- `liner` 可单独运行并生成两个完整 30 项线性公式及残差。
- `lowExp` 可基于指定 `liner` 结果单独运行，不重复线性拟合。
- `comb` 可用一条命令完成两个阶段并生成最终公式及解释文档。
- `comb` 可复用已有阶段产物，且能拒绝不匹配的运行结果。
- 最终线性公式中的 30 个变量全部出现，机器可读系数不被展示精度截断。
- 最终文档清楚区分模型空间和物理空间，不把编码后的 `tem/fcr/coldway` 误写成原始物理量。

## 10. 开发阶段划分

### 阶段 A：公共契约与数据校验

- 固化 30 维 CSV 列顺序和符号变量映射。
- 实现 CSV 与图节点特征的一致性校验。
- 统一目标头、mask、样本索引和 manifest 结构。
- 抽取或封装预处理脚本中的编解码接口。
- 为数据对齐和编解码建立单元测试。

### 阶段 B：实现 `liner`

- 建立强制 `liner` 目录和独立入口。
- 复用现有 RGAT 教师加载与前向逻辑。
- 实现 OLS/Ridge 双头线性蒸馏。
- 导出完整 30 项公式、系数、预测、残差和指标。
- 验证训练/验证隔离及线性公式回放一致性。

### 阶段 C：调整 `lowExp`

- 将输入目标改为 `liner` 产出的残差。
- 保留现有 SymTorch/PySR 搜索和公式导出能力。
- 增加 liner manifest 校验和残差指标。
- 调整文件名和文档措辞，明确输出为 residual equation。
- 使用小迭代 `--quick` 完成冒烟测试，再进行正式搜索测试。

### 阶段 D：实现 `comb`

- 建立强制 `comb` 目录和独立联合入口。
- 以子模块 API 或受控进程方式依次调用 `liner`、`lowExp`。
- 实现阶段复用、manifest 指纹匹配和失败传播。
- 组合两个预测头的线性与残差公式。
- 生成最终 JSON、模型空间 Markdown、物理空间 Markdown、summary 和指标。

### 阶段 E：端到端验证

- 使用 UTS/FS checkpoint 和 `datagnnUts.csv` 完成一次端到端运行。
- 使用 YS/FS checkpoint 完成目标切换验证。
- 对比原始 `lowExp`、仅线性模型、线性加残差公式三者的验证误差和公式复杂度。
- 随机抽取样本，人工核对 CSV 输入、图特征、线性输出、残差输出、组合输出和物理解码。
- 确认失败运行不会留下可被误认为成功结果的 `latest` 或完整标记。

## 11. 风险与处理原则

- **图信息无法由单节点 30 维输入完全恢复**：最终公式是 RGAT 的无图近似，必须报告对教师的验证蒸馏误差。
- **30 项出现不等于 30 项有效**：保留全部线性项，但解释时同时报告系数大小，禁止把结构性保留表述为统计显著性。
- **共线性导致系数不稳定**：比较 OLS 与 Ridge 的验证表现，并记录 alpha；不得为追求变量出现而使用伪造非零系数。
- **CSV 与图节点错位**：通过数值和索引校验阻断，而不是依赖默认行顺序。
- **验证泄漏**：线性参数和 PySR 搜索只使用训练 mask。
- **公式定义域错误**：对 train、val、全体样本及预先规定的物理边界探针执行有限值检查，任何 `NaN/Inf` 都阻断成功产物。
- **公式展示精度损失**：机器 JSON 保存高精度系数和原始表达式，Markdown 仅作展示。
- **coldway 编码不完全可逆**：明确 `[0,0]` 歧义，物理解释采用条件代换，不生成虚假的唯一逆解。
- **FS 存在 100 倍刻度差异**：所有产物记录目标刻度和单位，最终解释同时给出换算关系。
- **阶段结果误组合**：使用 manifest 指纹绑定 CSV、图数据、checkpoint、目标头、特征顺序、mask 和 seed。

## 12. 完成定义

只有同时满足以下条件，开发任务才视为完成：

1. `liner`、`lowExp`、`comb` 三个目录及职责边界按本计划落地。
2. `liner` 生成包含全部 30 维变量的双头线性蒸馏公式。
3. `lowExp` 生成双头非线性残差公式，而不是重新拟合完整教师输出。
4. `comb` 一键组合两部分，生成可回放的最终公式。
5. 最终公式同时提供模型空间版本和依据 `preprocess_datagnn_repro.py` 的人类可读编解码说明。
6. 所有阶段通过数据对齐、训练/验证隔离、公式回放和物理解码测试。
7. 最终报告能够清晰比较 RGAT 教师、线性部分、非线性残差部分和联合公式的精度与复杂度。
