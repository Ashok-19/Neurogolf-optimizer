from __future__ import annotations

from importlib import import_module

FAMILY_MODULES = [
    "identity", "crop", "bbox", "recolor", "object_extract", "connected_components",
    "line_draw", "symmetry", "tiling", "scaling", "pattern_fill", "template_match",
    "ray_projection", "rectangle", "sprite", "color_count", "mask_logic", "border",
    "gravity", "copy_paste", "arithmetic_grid",
]


def generate_family_programs(family: str, pairs, features, config) -> list:
    mod = import_module(f"{__package__}.{family}")
    return list(getattr(mod, "generate")(pairs, features, config))


def all_family_programs(pairs, features, config, families: list[str] | None = None) -> list:
    names = families or FAMILY_MODULES
    out = []
    for name in names:
        if name in FAMILY_MODULES:
            out.extend(generate_family_programs(name, pairs, features, config))
    return out
