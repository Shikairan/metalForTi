# Final equations (physical interpretation)

完整模型公式仍在模型空间求值。以下为输入/输出编解码规则（依据 preprocess_datagnn_repro.py）。

## Codec stats

```json
{
  "T_DIV": 800.0,
  "testenv_mean": [
    194.7433774834437,
    1342.7152317880796
  ],
  "testenv_std": [
    214.62536752046483,
    1547.0319516107018
  ],
  "mean_YS": 965.7821034430465,
  "mean_FS_dataOri2": 28.120464644701983,
  "mean_UTS": 1291.9939771048014,
  "FS_DATA1123_TO_DATAORI2_SCALE": 100.0
}
```

## Notes

- 元素 10 维保持 wt% 原值进入模型空间。
- testenv: tem/fcr 为 z-score；人类输入中 data1123 的 sr 与 fcr 同源管线，不能只改变量名。
- coldway: active * ln(T/800.0) 与 active * ln(t)；[0,0] 可能表示未激活或 T=800、t=1，不可无歧义逆推。
- YS_MPa = YS_model * mean_YS；UTS_MPa = UTS_model * mean_UTS；FS_dataOri2 = FS_model * mean_FS；FS_data1123 = FS_dataOri2 / 100。

## Model-space formulas (do not treat tem/fcr/coldway as raw physical fields)

```text
UTS = fcr^2 * tem * (0.10387048 - 0.097857766 * Zr) * (Mo + el_Si) / (Al + 0.08873339) + (0.040941132 * Al + 0.01924617 * Cr + 0.0061799585 * Fe - 0.0061552604 * Mo - 0.0064078769 * Nb + 0.061819415 * Sn - 0.15113067 * Ta - 0.0068071535 * V - 0.061953447 * Zr + 0.12368723 * coldway_0 + 0.081631698 * coldway_1 - 0.60072438 * coldway_10 - 0.027783996 * coldway_11 + 0.0017458583 * coldway_12 - 0.020902644 * coldway_13 - 0.42680002 * coldway_14 - 0.048078721 * coldway_15 + 1e - 06 * coldway_16 + 1e - 06 * coldway_17 - 0.18830134 * coldway_2 + 0.014543702 * coldway_3 - 0.26801711 * coldway_4 + 0.014773195 * coldway_5 - 1e - 06 * coldway_6 + 1e - 06 * coldway_7 - 0.10684088 * coldway_8 - 0.0069426307 * coldway_9 + 0.18028176 * el_Si + 0.031604143 * fcr - 0.1446218 * tem + 0.8350171)
FS = (0.3372095 * (tem - 0.5413826) * (tem + 0.65531564 / (fcr * (Al + fcr + 1.2845563))) - 0.2900931) + (-0.084648437 * Al - 0.008918471 * Cr - 0.037709053 * Fe - 0.0073925105 * Mo + 0.0070283025 * Nb + 0.018456156 * Sn - 0.19857633 * Ta - 0.00063631396 * V + 0.0029231156 * Zr - 1.6659319 * coldway_0 + 0.0005337073 * coldway_1 + 1.627312 * coldway_10 + 0.17515153 * coldway_11 + 0.021574481 * coldway_12 - 0.25830488 * coldway_13 + 0.03698807 * coldway_14 + 0.089631745 * coldway_15 + 1e - 06 * coldway_16 + 1e - 06 * coldway_17 - 0.55749016 * coldway_2 - 0.035941924 * coldway_3 + 0.027472595 * coldway_4 - 0.0078404365 * coldway_5 - 1e - 06 * coldway_6 - 1e - 06 * coldway_7 - 0.43179999 * coldway_8 - 0.072412147 * coldway_9 + 0.29563924 * el_Si - 0.13409522 * fcr + 0.16405049 * tem + 1.4511466)
```

coldway 编码不完全可逆：不生成虚假唯一逆解。
