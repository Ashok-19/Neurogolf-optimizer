from __future__ import annotations

from ..dsl import Program, Step


def generate(pairs, features, config):
    for color in range(1, 10):
        for row in range(30):
            yield Program(f"hline_{color}_{row}", "line_draw", {"color": color, "row": row}, [Step("Input"), Step("DrawLine", {"color": color, "row": row})])
        for col in range(30):
            yield Program(f"vline_{color}_{col}", "line_draw", {"color": color, "col": col}, [Step("Input"), Step("DrawLine", {"color": color, "col": col})])
        yield Program(f"diag_main_{color}", "line_draw", {"color": color}, [Step("Input"), Step("DrawLine", {"color": color, "r0": 0, "c0": 0, "r1": 29, "c1": 29})])
        yield Program(f"diag_anti_{color}", "line_draw", {"color": color}, [Step("Input"), Step("DrawLine", {"color": color, "r0": 0, "c0": 29, "r1": 29, "c1": 0})])
