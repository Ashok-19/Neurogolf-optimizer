from __future__ import annotations

from ..dsl import Program, Step
from ..grid import shape


def _global_map(pair):
    if shape(pair.input) != shape(pair.output):
        return None
    mapping = {}
    for r, row in enumerate(pair.input):
        for c, v in enumerate(row):
            out = pair.output[r][c] if r < len(pair.output) and c < len(pair.output[r]) else 0
            if v in mapping and mapping[v] != out:
                return None
            mapping[v] = out
    return mapping


def generate(pairs, features, config):
    maps = [_global_map(p) for p in pairs]
    maps = [m for m in maps if m is not None]
    if maps:
        keys = set.union(*(set(m) for m in maps))
        merged = {}
        ok = True
        for k in keys:
            vals = {m.get(k, k) for m in maps}
            if len(vals) != 1:
                ok = False
                break
            merged[k] = next(iter(vals))
        if ok and any(k != v for k, v in merged.items()):
            yield Program("global_recolor", "recolor", {"map": merged}, [Step("Input"), Step("Recolor", {"map": merged})])
    for old in range(10):
        for new in range(10):
            if old != new:
                yield Program(f"replace_{old}_to_{new}", "recolor", {"old": old, "new": new}, [Step("Input"), Step("ReplaceColor", {"old": old, "new": new})])
