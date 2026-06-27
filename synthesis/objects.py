from __future__ import annotations

from dataclasses import dataclass
from collections import deque

from .grid import bounding_box, colors, get, height, width


@dataclass(frozen=True)
class GridObject:
    color: int | None
    colors: set[int]
    pixels: frozenset[tuple[int, int]]
    bbox: tuple[int, int, int, int]
    height: int
    width: int
    area: int
    perimeter: int
    centroid: tuple[float, float]
    touches_border: bool
    is_rectangle: bool
    is_line: bool
    is_square: bool
    holes: int
    symmetry: set[str]
    dominant_color: int


def _neighbors(r, c, connectivity):
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dr, dc in dirs:
        yield r + dr, c + dc


def _features(grid, pix: set[tuple[int, int]]) -> GridObject:
    rs = [p[0] for p in pix]
    cs = [p[1] for p in pix]
    bbox = min(rs), min(cs), max(rs) + 1, max(cs) + 1
    h = bbox[2] - bbox[0]
    w = bbox[3] - bbox[1]
    vals = [get(grid, r, c) for r, c in pix]
    cs_set = set(vals)
    color = vals[0] if len(cs_set) == 1 else None
    perim = 0
    for r, c in pix:
        perim += sum((nr, nc) not in pix for nr, nc in _neighbors(r, c, 4))
    full_rect = h * w == len(pix)
    sym = set()
    rel = {(r - bbox[0], c - bbox[1]) for r, c in pix}
    if {(r, w - 1 - c) for r, c in rel} == rel:
        sym.add("h")
    if {(h - 1 - r, c) for r, c in rel} == rel:
        sym.add("v")
    counts = {v: vals.count(v) for v in cs_set}
    dominant = max(counts, key=counts.get)
    return GridObject(
        color=color,
        colors=cs_set,
        pixels=frozenset(pix),
        bbox=bbox,
        height=h,
        width=w,
        area=len(pix),
        perimeter=perim,
        centroid=(sum(rs) / len(rs), sum(cs) / len(cs)),
        touches_border=min(rs) == 0 or min(cs) == 0 or max(rs) == height(grid) - 1 or max(cs) == width(grid) - 1,
        is_rectangle=full_rect,
        is_line=h == 1 or w == 1 or len(pix) == max(h, w),
        is_square=h == w and full_rect,
        holes=max(0, h * w - len(pix)) if full_rect else 0,
        symmetry=sym,
        dominant_color=dominant,
    )


def extract_objects(grid, background: int | None = None, connectivity: int = 4) -> list[GridObject]:
    h, w = height(grid), width(grid)
    if background is None:
        background = 0
    seen: set[tuple[int, int]] = set()
    objs: list[GridObject] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen or get(grid, r, c) == background:
                continue
            color = get(grid, r, c)
            pix = set()
            q = deque([(r, c)])
            seen.add((r, c))
            while q:
                p = q.popleft()
                pix.add(p)
                for nr, nc in _neighbors(*p, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and get(grid, nr, nc) != background:
                        seen.add((nr, nc))
                        q.append((nr, nc))
            objs.append(_features(grid, pix))
    return objs


def extract_color_objects(grid, connectivity: int = 4) -> list[GridObject]:
    out = []
    for color in sorted(colors(grid) - {0}):
        masked = [[v if v == color else 0 for v in row] for row in grid]
        out.extend(extract_objects(masked, background=0, connectivity=connectivity))
    return out
