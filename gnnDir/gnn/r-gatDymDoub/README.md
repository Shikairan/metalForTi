# YS + FS dual-head RGAT (`SingleEncoder_DualRGAT` / alias `RGAT_Dual`)

This folder trains a **shared linear encoder + two independent RGAT stacks** (`RGATConv` + `edge_type`) for **joint YS and FS** node regression on the same graph bundle as R-GCN / r-gatFS:

- `.../gnnDir/gnndataPT/r-gatPT/material_graph.pt`
- `.../gnnDir/gnndataPT/r-gatPT/ys.pt`
- `.../gnnDir/gnndataPT/r-gatPT/fs.pt`
- `.../gnnDir/gnndataPT/r-gatPT/train_mask.pt`
- `.../gnnDir/gnndataPT/r-gatPT/val_mask.pt`

## Model

- Implementation: [`model_gat_double.py`](model_gat_double.py) (`SingleEncoder_DualRGAT`); [`model_gat.py`](model_gat.py) exports it and **`RGAT_Dual`** as an alias.
- **Encoder** on raw `x`, then **separate** YS and FS branches (each: 2× `RGATConv` with second layer `out_channels=hidden_dim`, `heads=1`, + LayerNorm + GELU + MLP head).

## Training loss and metrics

- **`train_fs_gat.py`:** each epoch does **two optimizer steps**: (1) forward `ys_pred, _ = model(...)`, **MSE** on `ys` over `train_mask`; (2) forward `_, fs_pred = model(...)`, **MSE** on `fs` over `train_mask`. No weighted combined loss.
- **Best checkpoint:** minimizes **`val_mae_fs + val_mae_ys`** (validation MAE from `_eval_metrics`).
- **`loop_train_swap_rgat.py`:** one **forward** per epoch; **dynamic priority** backward (dym.md): each step backprops **either** L1(YS) **or** L1(FS) by comparing `val_mae / hist_best` from the previous eval. Best checkpoint still by **`val_mae_fs + val_mae_ys`**. Curate/swap uses **difficulty-weighted** comb rel%: `w ∝ current_val_mae / hist_best_val_mae` per task (see `dym.md`).
- **FS diagnostics** (half/worst relative % in “raw” space) still use `_inverse_fs` (`exp`), same as the single-task FS scripts—apply to YS only if your labels share that convention.

## Run (standalone)

`train_fs_gat.py` **`--out-dir` must be under** this package’s `runs/` (same as the loop); default is `r-gatDymDoub/runs`.

```bash
mkdir -p runs
python3 train_fs_gat.py \
  --data-dir /home/data/symbolTransformer/symtest/gnnDir/gnndataPT/r-gatPT \
  --epochs 1000 \
  --lr 1e-3 \
  --hidden-dim 64 \
  --weight-decay 1e-4 \
  --dropout 0.2 \
  --seed 42 \
  --out-dir /home/data/symbolTransformer/symtest/gnnDir/gnn/r-gatDymDoub/runs
```

By default, `train_fs_gat.py` allows **inactive** nodes. Use **`--no-allow-inactive`** if masks must cover every node.

## Outputs

- **`best_ysfs_gat.pt`**: best checkpoint (metadata includes `model: "RGAT_Dual"`, `model_class`, `gat_heads`, `edge_dim`). Older checkpoints from other architectures are **not** compatible.
- **`train_log.csv`**: per-epoch FS/YS MAEs, `val_score`, and FS relative-error diagnostics.

## Mask loop (curate + exchange)

[`loop_train_swap_rgat.py`](loop_train_swap_rgat.py) trains each round with **difficulty-weighted** comb rel% for mask moves (weights from `val_mae` vs **historical best** `val_mae` per task, `dym.md`). **Checkpoint / state / history CSV** must live under this folder’s **`runs/`** (enforced). Graph tensors still load from **`--data-dir`**; masks are still written back to `data-dir`.

- Default **`--out-dir`**: `gnn/r-gatDymDoub/runs` (`best_ysfs_gat.pt`, `gat_mask_loop_state.json`, `gat_mask_swap_history.csv`).
- History CSV includes **`train_mae_fs` / `val_mae_fs` / `train_mae_ys` / `val_mae_ys`** (new files only; existing CSVs are not migrated).

```bash
cd /home/data/symbolTransformer/symtest/gnnDir/gnn/r-gatDymDoub
python3 loop_train_swap_rgat.py \
  --data-dir /home/data/symbolTransformer/symtest/gnnDir/gnndataPT/r-gatPT \
  --epochs-per-round 60 \
  --out-dir /home/data/symbolTransformer/symtest/gnnDir/gnn/r-gatDymDoub/runs
```

If an incompatible checkpoint is present, the loop **warns and trains from scratch** for that round until you remove or replace `best_ysfs_gat.pt`.

Use `--max-rounds N` to stop after N rounds, or Ctrl+C. See `--swap-batch-size`, `--max-curate`, `--log-interval` in `--help`.
