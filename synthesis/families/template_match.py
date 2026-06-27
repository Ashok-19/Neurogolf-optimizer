from __future__ import annotations

from ..dsl import Program, Step


def generate(pairs, features, config):
    for color in range(1, 10):
        yield Program(f"extract_template_color_{color}", "template_match", {"color": color}, [Step("Input"), Step("CropColorBBox", {"color": color, "output_h": 30, "output_w": 30})])
    if pairs:
        yield Program("paste_first_output_as_sprite", "template_match", {"object": pairs[0].output}, [Step("Input"), Step("PasteObject", {"object": pairs[0].output})])
