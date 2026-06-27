from __future__ import annotations

from ..dsl import Program, Step


def generate(pairs, features, config):
    yield Program("flip_h", "symmetry", steps=[Step("Input"), Step("Flip", {"axis": "h"})])
    yield Program("flip_v", "symmetry", steps=[Step("Input"), Step("Flip", {"axis": "v"})])
    yield Program("rotate90", "symmetry", steps=[Step("Input"), Step("Rotate", {"deg": 90})])
    yield Program("rotate180", "symmetry", steps=[Step("Input"), Step("Rotate", {"deg": 180})])
    yield Program("rotate270", "symmetry", steps=[Step("Input"), Step("Rotate", {"deg": 270})])
