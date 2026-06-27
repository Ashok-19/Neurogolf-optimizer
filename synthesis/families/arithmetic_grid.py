from __future__ import annotations

from ..dsl import Program, Step


def generate(pairs, features, config):
    yield Program("count_identity", "arithmetic_grid", steps=[Step("Input")])
    for color in range(1, 10):
        yield Program(f"count_mask_{color}", "arithmetic_grid", {"color": color}, [Step("Input"), Step("SelectColor", {"color": color})])
