from __future__ import annotations

from ..dsl import Program, Step


def generate(pairs, features, config):
    for conn in (4, 8):
        yield Program(f"largest_object_c{conn}", "connected_components", {"connectivity": conn}, [Step("Input"), Step("LargestObject", {"connectivity": conn})])
        yield Program(f"smallest_object_c{conn}", "connected_components", {"connectivity": conn}, [Step("Input"), Step("SmallestObject", {"connectivity": conn})])
    for color in range(1, 10):
        yield Program(f"object_by_color_{color}", "connected_components", {"color": color}, [Step("Input"), Step("ObjectByColor", {"color": color})])
