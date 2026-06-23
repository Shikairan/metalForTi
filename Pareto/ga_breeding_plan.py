"""拓展育种子代数量拆分（无 torch 依赖）。"""

from __future__ import annotations

DEFAULT_IMMIGRANT_FRAC = 0.10
DEFAULT_RANDOM_MATE_FRAC = 0.20
DEFAULT_VIRTUAL_BREED_SAMPLE = 200
IMMIGRANT_MUT_SCALE = 1.5


def split_offspring_counts(
    pop_size: int,
    *,
    immigrant_frac: float = DEFAULT_IMMIGRANT_FRAC,
    random_mate_frac: float = DEFAULT_RANDOM_MATE_FRAC,
) -> tuple[int, int, int]:
    """返回 (n_tournament, n_random_mate, n_immigrant)，三者之和为 pop_size。"""
    n_immigrant = int(round(pop_size * immigrant_frac))
    n_immigrant = max(0, min(n_immigrant, pop_size))
    n_cross = pop_size - n_immigrant
    n_random = int(round(n_cross * random_mate_frac))
    n_random = max(0, min(n_random, n_cross))
    n_tournament = n_cross - n_random
    return n_tournament, n_random, n_immigrant
