from __future__ import annotations

from ..dsl import Program, Step


def generate(pairs, features, config):
    for color in range(1, 10):
        yield Program(f"mask_keep_color_{color}", "mask_logic", {"color": color}, [Step("Input"), Step("CropColorBBox", {"color": color})])
