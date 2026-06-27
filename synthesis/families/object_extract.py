from __future__ import annotations

from ..dsl import Program, Step


def generate(pairs, features, config):
    # Conservative object extraction: crop the bounding box of a selected color.
    for color in range(1, 10):
        yield Program(f"extract_color_{color}", "object_extract", {"color": color}, [Step("Input"), Step("CropColorBBox", {"color": color})])
