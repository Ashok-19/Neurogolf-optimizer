from __future__ import annotations

from ..dsl import Program, Step


def generate(pairs, features, config):
    yield Program("copy_input", "identity", steps=[Step("Input")])
    for color in range(10):
        yield Program(f"remove_color_{color}", "identity", steps=[Step("Input"), Step("ReplaceColor", {"old": color, "new": 0})])
