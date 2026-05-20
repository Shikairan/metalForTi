#!/usr/bin/env bash
# 初学者：一键按顺序跑 highExp → medExp → lowExp → sampleExp
# 第一次请先单独跑：  python3.13 lowExp/run_distill.py --quick
# 自检：            python3.13 scripts/check_env.py
# 教程：            阅读 README.md
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
