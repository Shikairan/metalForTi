# FS-only GAT Training

This folder trains a **GAT** model for **FS regression only** using the same graph bundle as R-GCN:

- `.../gnnDir/gnndataPT/r-gatPT/material_graph.pt`
- `.../gnnDir/gnndataPT/r-gatPT/fs.pt`
- `.../gnnDir/gnndataPT/r-gatPT/train_mask.pt`
- `.../gnnDir/gnndataPT/r-gatPT/val_mask.pt`

## Model

- Two **`RGATConv`** layers: first layer multi-head (`hidden_dim` divisible by head count), second layer `heads=1` with full `hidden_dim` output so **`h1 + h2` residual** matches dimension. Relation ids `0/1/2` are passed as **`edge_type`** (`comp_sim` / `env_sim` / `heat_sim`), not one-hot `edge_attr`.
- LayerNorm, GELU, dropout, residual, then MLP head `Linear → GELU → Dropout → Linear` to scalar FS.

## Run

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
  --out-dir /home/data/symbolTransformer/symtest/gnnDir/gnn/r-gat/runs
```

By default, `train_fs_gat.py` allows **inactive** nodes (neither train nor val), e.g. after mask curate loops. Use **`--no-allow-inactive`** if your masks must cover every node.

## Outputs

- `best_fs_gat.pt`: best checkpoint by validation MAE (includes `model: "FSGAT"`, `gat_heads`, `edge_dim`, `backbone: "RGATConv"`). Old checkpoints from `GATConv`+`edge_attr` are **not** load-compatible; **`loop_train_swap_rgat.py` will warn and train from scratch** until you remove or overwrite that file.
- `train_log.csv`: per-epoch metrics (same columns as `train_fs_rgcn.py`).

## Mask loop (curate + exchange)

[`loop_train_swap_rgat.py`](loop_train_swap_rgat.py) mirrors `r-gnn/loop_train_swap_rgcn.py` but trains **FSGAT** (`RGATConv` + `edge_type`). Default `--out-dir` in script points to **`gnn/r-gat/runs`** (`best_fs_gat.pt`, `gat_mask_loop_state.json`, `gat_mask_swap_history.csv`). It **overwrites** `--data-dir/train_mask.pt` and `val_mask.pt` each round.

```bash
cd /home/data/symbolTransformer/symtest/gnnDir/gnn/r-gatFS
python3 loop_train_swap_rgat.py \
  --data-dir /home/data/symbolTransformer/symtest/gnnDir/gnndataPT/r-gatPT \
  --epochs-per-round 60 \
  --out-dir /home/data/symbolTransformer/symtest/gnnDir/gnn/r-gat/runs
```

Use `--max-rounds N` to stop after N rounds, or Ctrl+C. See `--swap-batch-size`, `--max-curate`, `--log-interval` in script help.
