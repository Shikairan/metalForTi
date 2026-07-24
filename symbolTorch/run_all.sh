#!/usr/bin/env bash
# lowExp 蒸馏（每次结果写入 runs/<时间戳>/，不清理历史）
set -uo pipefail

PY="${PYTHON:-/root/miniconda3/envs/metal/bin/python}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
CKPT="${CKPT:-$ROOT/../modelAll/ysFs/runs/best_rgat_full.pt}"
EXTRA=("$@")

mkdir -p "$ROOT/logs" "$ROOT/lowExp/runs"
rm -f "$ROOT/runs_all.log"
rm -f "$ROOT/logs/"*.log

run_one() {
  local dir="$1"
  shift
  echo ">>>>>>>>>> $dir 开始 $(date -u)" | tee -a "$ROOT/runs_all.log"
  if (cd "$ROOT/$dir" && "$PY" run_distill.py --quick --device cpu --ckpt "$CKPT" "$@" "${EXTRA[@]}" > "$ROOT/logs/${dir}.log" 2>&1); then
    echo ">>>>>>>>>> $dir 完成 $(date -u)" | tee -a "$ROOT/runs_all.log"
    if [[ -L "$ROOT/$dir/runs/latest" ]]; then
      echo ">>>>>>>>>> 输出 → $ROOT/$dir/runs/$(readlink "$ROOT/$dir/runs/latest")" | tee -a "$ROOT/runs_all.log"
    fi
    return 0
  fi
  echo ">>>>>>>>>> $dir 失败 $(date -u)，详见 logs/${dir}.log" | tee -a "$ROOT/runs_all.log"
  return 1
}

echo "=== symbolTorch 批量开始 $(date -u) ===" | tee "$ROOT/runs_all.log"
echo "PY=$PY CKPT=$CKPT（历史 runs 子目录保留，不删除）" | tee -a "$ROOT/runs_all.log"

run_one lowExp || true

echo "[OK] All finished $(date -u)" | tee -a "$ROOT/runs_all.log"
