BASE=/home/data/metalTi/metalForTi/gnnDir
RUNS=$BASE/gnn/r-gatDouble/runs

# 1) 删掉 Double 侧所有 loop / 训练残留
rm -f "$RUNS/gat_mask_loop_state.json"
rm -f "$RUNS/gat_mask_swap_history.csv"
rm -f "$RUNS/best_ysfs_gat.pt"
rm -f "$RUNS/train_log.csv"
rm -rf "$RUNS/mask_round_history"
rm -rf "$RUNS/mask_val_lt10"

# 2) 重新生成 r-gat PT（图 + ys/fs + 随机 8:2 mask，无 inactive）
cd "$BASE"
python3 regenerate_rgnnpt.py \
  --pt-bundle rgat \
  --skip-build-datagnn \
  --datagnn-csv datacsv/datagnn.csv \
  --train-ratio 0.800 \
  --split-seed 19950217
