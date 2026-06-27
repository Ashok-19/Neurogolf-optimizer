from __future__ import annotations

from ..dsl import Program, Step


def generate(pairs, features, config):
    for color in range(1, 10):
        yield Program(f"fill_bbox_{color}", "pattern_fill", {"target_color": color}, [Step("Input"), Step("FillRectangle", {"target_color": color, "fill_color": color})])
        yield Program(f"draw_bbox_{color}", "pattern_fill", {"target_color": color}, [Step("Input"), Step("DrawRectangle", {"target_color": color, "draw_color": color})])
    if pairs:
        tile = pairs[0].output[: min(3, len(pairs[0].output))]
        yield Program("tile_first_output_prefix", "pattern_fill", {"tile": tile}, [Step("Input"), Step("Tile", {"tile": tile, "height": 30, "width": 30})])
