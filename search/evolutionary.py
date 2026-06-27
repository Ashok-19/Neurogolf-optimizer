from __future__ import annotations

import random

from ..models import Mutation
from .beam import compatible_mutations


def sample_mutation_sets(mutations: list[Mutation], max_size: int, trials: int, seed: int) -> list[list[Mutation]]:
    rng = random.Random(seed)
    if not mutations:
        return []
    out = []
    weighted = sorted(mutations, key=lambda m: m.priority, reverse=True)
    attempts = 0
    while len(out) < trials and attempts < trials * 20:
        attempts += 1
        size = rng.randint(1, max(1, min(max_size, len(weighted))))
        pool = weighted[: min(len(weighted), max(20, size * 8))]
        combo = rng.sample(pool, size)
        if compatible_mutations(combo):
            out.append(combo)
    return out
