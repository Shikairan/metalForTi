"""Unit tests for codec wrappers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.codec import (  # noqa: E402
    codec_stats_dict,
    decode_fs_model,
    decode_head0_model,
    model_to_physical_30,
)


def test_codec_stats_and_decode():
    stats = codec_stats_dict()
    assert "T_DIV" in stats
    assert stats["FS_DATA1123_TO_DATAORI2_SCALE"] == 100.0
    ys = decode_head0_model(1.0, head0_name="YS")
    assert "YS_MPa" in ys
    uts = decode_head0_model(1.0, head0_name="UTS")
    assert "UTS_MPa" in uts
    fs = decode_fs_model(1.0)
    assert abs(fs["FS_dataOri2"] / fs["FS_data1123"] - 100.0) < 1e-6

    g = np.zeros(30, dtype=np.float32)
    g[10] = 0.0
    g[11] = 0.0
    phys = model_to_physical_30(g)
    assert phys.shape == (30,)
