from __future__ import annotations

from ..dsl import Program, Step


def generate(pairs, features, config):
    for target in range(1, 10):
        for draw in range(1, 10):
            yield Program(f"draw_rect_{target}_{draw}", "rectangle", {"target_color": target, "draw_color": draw}, [Step("Input"), Step("DrawRectangle", {"target_color": target, "draw_color": draw})])
            yield Program(f"fill_rect_{target}_{draw}", "rectangle", {"target_color": target, "fill_color": draw}, [Step("Input"), Step("FillRectangle", {"target_color": target, "fill_color": draw})])
