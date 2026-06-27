from __future__ import annotations

from ..dsl import Program, Step


def generate(pairs, features, config):
    if pairs:
        for r, c in [(0, 0), (1, 1), (2, 2)]:
            yield Program(f"paste_first_output_{r}_{c}", "copy_paste", {"r": r, "c": c}, [Step("Input"), Step("PasteObject", {"object": pairs[0].output, "r": r, "c": c})])
    for color in range(1, 10):
        yield Program(f"copy_color_object_{color}", "copy_paste", {"color": color}, [Step("Input"), Step("ObjectByColor", {"color": color})])
