from __future__ import annotations

from dataclasses import dataclass, field

from .grid import bounding_box, color_counts, colors, grid_equal_with_missing_zero, shape
from .objects import extract_objects
from .task_data import GridPair


@dataclass
class PairFeatures:
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    input_colors: set[int]
    output_colors: set[int]
    new_colors: set[int]
    removed_colors: set[int]
    color_map_candidates: dict[int, int]
    crop_bbox: tuple[int, int, int, int] | None
    object_count: int
    tags: set[str] = field(default_factory=set)


@dataclass
class TaskFeatures:
    pairs: list[PairFeatures]
    tags: set[str]


def pair_features(pair: GridPair) -> PairFeatures:
    ic, oc = colors(pair.input), colors(pair.output)
    cmap = {}
    for old in ic:
        in_positions = [(r, c) for r, row in enumerate(pair.input) for c, v in enumerate(row) if v == old]
        outs = {pair.output[r][c] for r, c in in_positions if r < len(pair.output) and c < len(pair.output[r])}
        if len(outs) == 1:
            cmap[old] = next(iter(outs))
    bbox = None
    for color in ic - {0}:
        b = bounding_box(pair.input, color)
        if b is not None:
            cropped = [[pair.input[r][c] for c in range(b[1], b[3])] for r in range(b[0], b[2])]
            if grid_equal_with_missing_zero(cropped, pair.output):
                bbox = b
                break
    tags = set()
    if grid_equal_with_missing_zero(pair.input, pair.output):
        tags.add("identity")
    if shape(pair.input) == shape(pair.output) and cmap:
        tags.add("recolor")
    if bbox is not None:
        tags.add("crop")
        tags.add("bbox")
    if shape(pair.output)[0] in {shape(pair.input)[0] * 2, shape(pair.input)[0] * 3, shape(pair.input)[0] * 4}:
        tags.add("scaling")
    if len(extract_objects(pair.input)) > 0:
        tags.add("object_extract")
    return PairFeatures(shape(pair.input), shape(pair.output), ic, oc, oc - ic, ic - oc, cmap, bbox, len(extract_objects(pair.input)), tags)


def task_features(pairs: list[GridPair]) -> TaskFeatures:
    pfs = [pair_features(p) for p in pairs]
    if not pfs:
        return TaskFeatures([], {"unknown"})
    common = set.intersection(*(pf.tags for pf in pfs)) if all(pf.tags for pf in pfs) else set()
    tags = common or set.union(*(pf.tags for pf in pfs)) or {"unknown"}
    return TaskFeatures(pfs, tags)
