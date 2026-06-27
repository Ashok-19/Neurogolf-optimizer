from __future__ import annotations

from ..dsl import Program, Step


def generate(pairs, features, config):
    # Removing a uniform border is represented as static crop if detected by features.
    for pf in features.pairs:
        ih, iw = pf.input_shape
        oh, ow = pf.output_shape
        if ih == oh + 2 and iw == ow + 2:
            yield Program("remove_one_cell_border", "border", steps=[Step("Input"), Step("Crop", {"bbox": (1, 1, ih - 1, iw - 1)})])
            break
