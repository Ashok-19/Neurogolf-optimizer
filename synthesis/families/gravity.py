from __future__ import annotations

from ..dsl import Program, Step


def generate(pairs, features, config):
    for direction in ("down", "up", "left", "right"):
        yield Program(f"gravity_{direction}", "gravity", {"direction": direction}, [Step("Input"), Step("Gravity", {"direction": direction})])
