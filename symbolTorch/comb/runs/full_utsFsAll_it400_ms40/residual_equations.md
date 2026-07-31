# lowExp residual equations

## 说明

- 拟合目标为 teacher - linear（liner 残差），不是完整教师输出或真实标签。
- 推理仅使用节点 30 维特征，不使用图邻居。
- SLIME=False | SR niterations=400 maxsize=40
- liner_run=/home/data/metalgnn/metalForTi/symbolTorch/comb/runs/full_utsFsAll_it400_ms40/liner_stage
- liner_fingerprint=19881268f0e9667fcdf863d6fc60dfbb

## R_UTS

完整方程：

```text
R_UTS = fcr * tem * (fcr / (Al + 0.08873339)) * (0.10387048 - 0.097857766 * Zr) * (Mo + el_Si)
```

出现的变量：

| 符号 | 含义 |
|------|------|
| `el_Si` | 元素 Si 含量 wt% |
| `tem` | 试验温度（模型空间 / z-score，对应 data 中 tem） |
| `fcr` | 试验环境第二维（模型空间 / z-score；图特征名 fcr，与 data1123 的 sr 同源管线） |
| `Al` | 元素 Al 含量 wt% |
| `Zr` | 元素 Zr 含量 wt% |
| `Mo` | 元素 Mo 含量 wt% |

- block: `residual_uts`
- 原始 RHS: `fcr*tem*(fcr/(Al + 0.08873339))*(0.10387048 - 0.097857766*Zr)*(Mo + el_Si)`

## R_FS

完整方程：

```text
R_FS = (tem - 0.5413826) * (tem - (-1) * 0.65531564 / (fcr * (fcr - (-Al - 1.2845563)))) * 0.3372095 - 1 * 0.2900931
```

出现的变量：

| 符号 | 含义 |
|------|------|
| `tem` | 试验温度（模型空间 / z-score，对应 data 中 tem） |
| `fcr` | 试验环境第二维（模型空间 / z-score；图特征名 fcr，与 data1123 的 sr 同源管线） |
| `Al` | 元素 Al 含量 wt% |

- block: `residual_fs`
- 原始 RHS: `(tem - 0.5413826)*(tem - (-1)*0.65531564/(fcr*(fcr - (-Al - 1.2845563))))*0.3372095 - 1*0.2900931`

## 全部 30 维特征名（参考）

| 下标 | 符号 | 含义 |
|------|------|------|
| 0 | `Al` | 元素 Al 含量 wt% |
| 1 | `Zr` | 元素 Zr 含量 wt% |
| 2 | `Sn` | 元素 Sn 含量 wt% |
| 3 | `Mo` | 元素 Mo 含量 wt% |
| 4 | `Cr` | 元素 Cr 含量 wt% |
| 5 | `Nb` | 元素 Nb 含量 wt% |
| 6 | `el_Si` | 元素 Si 含量 wt% |
| 7 | `V` | 元素 V 含量 wt% |
| 8 | `Ta` | 元素 Ta 含量 wt% |
| 9 | `Fe` | 元素 Fe 含量 wt% |
| 10 | `tem` | 试验温度（模型空间 / z-score，对应 data 中 tem） |
| 11 | `fcr` | 试验环境第二维（模型空间 / z-score；图特征名 fcr，与 data1123 的 sr 同源管线） |
| 12 | `coldway_0` | 工艺 coldway 第 0 维（18 维展平，模型空间） |
| 13 | `coldway_1` | 工艺 coldway 第 1 维（18 维展平，模型空间） |
| 14 | `coldway_2` | 工艺 coldway 第 2 维（18 维展平，模型空间） |
| 15 | `coldway_3` | 工艺 coldway 第 3 维（18 维展平，模型空间） |
| 16 | `coldway_4` | 工艺 coldway 第 4 维（18 维展平，模型空间） |
| 17 | `coldway_5` | 工艺 coldway 第 5 维（18 维展平，模型空间） |
| 18 | `coldway_6` | 工艺 coldway 第 6 维（18 维展平，模型空间） |
| 19 | `coldway_7` | 工艺 coldway 第 7 维（18 维展平，模型空间） |
| 20 | `coldway_8` | 工艺 coldway 第 8 维（18 维展平，模型空间） |
| 21 | `coldway_9` | 工艺 coldway 第 9 维（18 维展平，模型空间） |
| 22 | `coldway_10` | 工艺 coldway 第 10 维（18 维展平，模型空间） |
| 23 | `coldway_11` | 工艺 coldway 第 11 维（18 维展平，模型空间） |
| 24 | `coldway_12` | 工艺 coldway 第 12 维（18 维展平，模型空间） |
| 25 | `coldway_13` | 工艺 coldway 第 13 维（18 维展平，模型空间） |
| 26 | `coldway_14` | 工艺 coldway 第 14 维（18 维展平，模型空间） |
| 27 | `coldway_15` | 工艺 coldway 第 15 维（18 维展平，模型空间） |
| 28 | `coldway_16` | 工艺 coldway 第 16 维（18 维展平，模型空间） |
| 29 | `coldway_17` | 工艺 coldway 第 17 维（18 维展平，模型空间） |
