from __future__ import annotations

from .crop import generate as crop_generate


def generate(pairs, features, config):
    yield from crop_generate(pairs, features, config)
