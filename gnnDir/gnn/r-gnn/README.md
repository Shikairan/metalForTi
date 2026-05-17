# FS-only R-GCN Training

This folder trains an R-GCN model for **FS regression only** using graph data from:

- `/home/data/symbolTransformer/symtest/gnnDir/gnndataPT/r-gnnPT/material_graph.pt`
- `/home/data/symbolTransformer/symtest/gnnDir/gnndataPT/r-gnnPT/fs.pt`
- `/home/data/symbolTransformer/symtest/gnnDir/gnndataPT/r-gnnPT/train_mask.pt`
- `/home/data/symbolTransformer/symtest/gnnDir/gnndataPT/r-gnnPT/val_mask.pt`

## Model

- `RGCNConv -> RGCNConv -> Linear(1)`
- 3 relation types:
  - `comp_sim` -> edge type 0
  - `env_sim` -> edge type 1
  - `heat_sim` -> edge type 2

## Stable preset run

```bash
python3 train_fs_rgcn.py \
  --data-dir /home/data/symbolTransformer/symtest/gnnDir/gnndataPT/r-gnnPT \
  --epochs 1000 \
  --lr 1e-3 \
  --hidden-dim 64 \
  --weight-decay 1e-4 \
  --dropout 0.2 \
  --seed 42 \
  --out-dir /home/data/symbolTransformer/symtest/gnnDir/gnn/r-gnn/runs
```

## Outputs

- `best_fs_rgcn.pt`: best checkpoint by validation MAE
- `train_log.csv`: per-epoch train/val MAE

