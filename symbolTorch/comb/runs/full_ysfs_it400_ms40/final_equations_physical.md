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
YS = (fcr * (fcr + tem + 0.33882254) - ((1.0123826 - (-0.42776537) * Nb * (coldway_2 - 1 * (-0.09725147))) * (tem - 0.29989803) - 1.8937974 / tem)) * (-0.056415386) + (-0.0045934531 * Al - 0.0037284806 * Cr - 0.015504235 * Fe - 0.0032740204 * Mo - 0.0079766208 * Nb + 0.0012867357 * Sn - 0.029698557 * Ta + 0.00014749239 * V - 0.0002791068 * Zr + 0.028295745 * coldway_0 - 0.0035485636 * coldway_1 + 0.00071516687 * coldway_10 - 0.0031596875 * coldway_11 - 0.0021395838 * coldway_12 + 0.025616604 * coldway_13 - 0.017070394 * coldway_14 + 0.0043642928 * coldway_15 - 0.014670133 * coldway_2 - 0.0058108537 * coldway_3 + 0.0099503173 * coldway_4 - 0.020956705 * coldway_5 - 0 * coldway_6 - 0 * coldway_7 + 0.018093365 * coldway_8 + 0.0048253464 * coldway_9 + 0.010598456 * el_Si + 0.10858769 * fcr - 0.20834024 * tem + 0 + 0 + 1.0643951)
FS = (-0.041649641 * Al - 0.0072575245 * Cr + 0.07106529 * Fe + 0.0069687014 * Mo + 0.0015118282 * Nb + 0.012232036 * Sn + 0.016112946 * Ta + 0.006388686 * V - 0.037676273 * Zr - 0.011618618 * coldway_0 - 0.011095486 * coldway_1 - 0.0025384681 * coldway_10 + 0.01362151 * coldway_11 + 0.0015757283 * coldway_12 - 0.018865729 * coldway_13 - 0.0036569742 * coldway_14 - 0.0010637431 * coldway_15 + 0.0092904458 * coldway_2 + 0.016098633 * coldway_3 - 0.011339343 * coldway_4 + 0.0051780967 * coldway_5 - 0 * coldway_6 + 0 * coldway_7 - 0.0094265406 * coldway_8 - 0.006187401 * coldway_9 - 0.021524373 * el_Si - 0.12758038 * fcr + 0.10028992 * tem + 0 + 0 + 1.109775) + sin(-(-0.30370873) * fcr + tem / (Zr + el_Si + fcr + 0.36150554)) * (-0.07616235) / (tem + 0.3605962)
```

coldway 编码不完全可逆：不生成虚假唯一逆解。
