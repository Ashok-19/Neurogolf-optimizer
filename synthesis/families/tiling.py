from __future__ import annotations

from ..dsl import Program, Step


def generate(pairs, features, config):
    if getattr(config, "allow_constant_output", False):
        for i, pair in enumerate(pairs[:8]):
            yield Program(f"constant_output_{i}", "tiling", {"grid": pair.output}, [Step("Constant", {"grid": pair.output})])
