#!/usr/bin/env python3
"""Quick environment check before running symbolTorch distillation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.constants import DEFAULT_CKPT, DEFAULT_DATA_DIR, FEATURE_NAMES, PYTHON_BIN_HINT  # noqa: E402


def main() -> int:
    ok = True
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 11):
        print(f"[WARN] torch-symbolic needs Python >= 3.11. Suggested: {PYTHON_BIN_HINT}")
        ok = False

    try:
        import torch
        import torch_geometric  # noqa: F401

        print(f"[OK] torch {torch.__version__}, torch_geometric")
    except ImportError as e:
        print(f"[FAIL] {e}")
        ok = False

    try:
        from symtorch import SymbolicModel  # noqa: F401

        print("[OK] symtorch (torch-symbolic)")
    except ImportError as e:
        print(f"[FAIL] symtorch: {e}")
        ok = False

    print(f"Features ({len(FEATURE_NAMES)}): {', '.join(FEATURE_NAMES[:5])} ...")
    print(f"Data dir: {DEFAULT_DATA_DIR} -> {'exists' if DEFAULT_DATA_DIR.is_dir() else 'MISSING'}")
    print(f"Teacher ckpt: {DEFAULT_CKPT} -> {'exists' if DEFAULT_CKPT.is_file() else 'MISSING'}")

    if not DEFAULT_DATA_DIR.joinpath("material_graph.pt").is_file():
        print("  Hint: cd metalForTi/gnnDir && python regenerate_rgnnpt.py --pt-bundle rgat")
        ok = False
    if not DEFAULT_CKPT.is_file():
        print("  Hint: cd metalForTi/gnnDir/gnn/r-gatDouble && python train_fs_gat.py ...")
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
