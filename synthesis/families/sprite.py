from __future__ import annotations

from ..dsl import Program, Step


def generate(pairs, features, config):
    for color in range(1, 10):
        yield Program(f"sprite_color_{color}", "sprite", {"color": color}, [Step("Input"), Step("ObjectByColor", {"color": color})])
    if pairs:
        yield Program("constant_first_sprite", "sprite", {"grid": pairs[0].output}, [Step("Constant", {"grid": pairs[0].output})])
