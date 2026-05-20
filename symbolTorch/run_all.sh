#!/usr/bin/env bash
# Run all four symbolTorch distillation experiments.
set -euo pipefail

PY="${PYTHON:-/root/miniconda3/bin/python3.13}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
EXTRA=("$@")

run_one() {
  local dir="$1"
  echo ""
  echo ">>>>>>>>>> $dir <<<<<<<<<<"
  (cd "$ROOT/$dir" && "$PY" run_distill.py "${EXTRA[@]}")
}

run_one highExp
run_one medExp
run_one lowExp
run_one sampleExp

echo ""
echo "[OK] All four experiments finished. See */runs/metrics.json"
