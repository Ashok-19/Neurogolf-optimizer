from __future__ import annotations

from ..dsl import Program, Step


def generate(pairs, features, config):
    for factor in (2, 3, 4):
        yield Program(f"scale_{factor}", "scaling", {"factor": factor}, [Step("Input"), Step("Scale", {"factor": factor})])
