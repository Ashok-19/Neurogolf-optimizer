from __future__ import annotations

from ..dsl import Program, Step


def generate(pairs, features, config):
    for color in range(1, 10):
        for dr, dc, name in [(0, 1, "right"), (0, -1, "left"), (1, 0, "down"), (-1, 0, "up")]:
            yield Program(f"ray_{name}_{color}", "ray_projection", {"color": color, "dr": dr, "dc": dc}, [Step("Input"), Step("DrawRay", {"color": color, "r": 0, "c": 0, "dr": dr, "dc": dc})])
