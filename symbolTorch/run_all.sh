#!/usr/bin/env bash
# 清理 → 四档蒸馏 → 汇总
set -uo pipefail

PY="${PYTHON:-/root/miniconda3/envs/metal/bin/python}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
CKPT="${CKPT:-$ROOT/../modelAll/runs/best_rgat_full.pt}"
EXTRA=("$@")

echo "=== 清理旧结果 ==="
for d in highExp medExp lowExp sampleExp; do
  if [[ -d "$ROOT/$d/runs" ]]; then
    find "$ROOT/$d/runs" -mindepth 1 ! -name '.gitkeep' -delete 2>/dev/null || true
  fi
done
mkdir -p "$ROOT/logs"
rm -f "$ROOT/runs_all.log" "$ROOT/SYMBOLTORCH_SUMMARY.json" "$ROOT/SYMBOLTORCH_SUMMARY.md"
rm -f "$ROOT/logs/"*.log

run_one() {
  local dir="$1"
  shift
  echo ">>>>>>>>>> $dir 开始 $(date -u)" | tee -a "$ROOT/runs_all.log"
  if (cd "$ROOT/$dir" && "$PY" run_distill.py --quick --device cpu --ckpt "$CKPT" "$@" "${EXTRA[@]}" > "$ROOT/logs/${dir}.log" 2>&1); then
    echo ">>>>>>>>>> $dir 完成 $(date -u)" | tee -a "$ROOT/runs_all.log"
    return 0
  fi
  echo ">>>>>>>>>> $dir 失败 $(date -u)，详见 logs/${dir}.log" | tee -a "$ROOT/runs_all.log"
  return 1
}

echo "=== symbolTorch 批量开始 $(date -u) ===" | tee "$ROOT/runs_all.log"
echo "PY=$PY CKPT=$CKPT" | tee -a "$ROOT/runs_all.log"

run_one highExp || true
run_one medExp --encoder-sym-dir "$ROOT/highExp/runs" || true
run_one lowExp || true
run_one sampleExp --top-k 1 || true

echo "=== 生成汇总 $(date -u) ===" | tee -a "$ROOT/runs_all.log"
(cd "$ROOT" && "$PY" scripts/build_summary.py --ckpt "$CKPT" --quick) | tee -a "$ROOT/runs_all.log"

echo "[OK] All finished $(date -u)" | tee -a "$ROOT/runs_all.log"
