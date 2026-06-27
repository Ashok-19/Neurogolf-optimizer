from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

BBox = tuple[int, int, int, int]


def height(grid): return len(grid or [])
def width(grid): return max((len(r) for r in (grid or [])), default=0)
def shape(grid): return height(grid), width(grid)


def get(grid, r: int, c: int, default: int = 0) -> int:
    return grid[r][c] if 0 <= r < len(grid) and 0 <= c < len(grid[r]) else default


def to_numpy(grid, fill: int = 0) -> np.ndarray:
    h, w = shape(grid)
    arr = np.full((h, w), fill, dtype=np.int64)
    for r, row in enumerate(grid or []):
        for c, v in enumerate(row):
            arr[r, c] = int(v)
    return arr


def from_numpy(arr: np.ndarray) -> list[list[int]]:
    return [[int(x) for x in row] for row in np.asarray(arr).tolist()]


def colors(grid) -> set[int]:
    return {int(v) for row in (grid or []) for v in row}


def color_counts(grid) -> Counter:
    return Counter(int(v) for row in (grid or []) for v in row)


def bounding_box(mask_or_grid, color: int | None = None) -> BBox | None:
    arr = to_numpy(mask_or_grid)
    if arr.size == 0:
        return None
    mask = arr == color if color is not None else arr.astype(bool)
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1) + 1, int(c1) + 1


def crop(grid, bbox: BBox | None):
    if bbox is None:
        return []
    r0, c0, r1, c1 = bbox
    return [[get(grid, r, c) for c in range(c0, c1)] for r in range(r0, r1)]


def pad_to_30(grid, fill: int = 0):
    arr = np.full((30, 30), fill, dtype=np.int64)
    src = to_numpy(grid, fill=fill)
    h, w = src.shape if src.ndim == 2 else (0, 0)
    arr[: min(h, 30), : min(w, 30)] = src[:30, :30]
    return from_numpy(arr)


def rotate90(grid): return from_numpy(np.rot90(to_numpy(grid), 1))
def rotate180(grid): return from_numpy(np.rot90(to_numpy(grid), 2))
def rotate270(grid): return from_numpy(np.rot90(to_numpy(grid), 3))
def flip_h(grid): return from_numpy(np.fliplr(to_numpy(grid)))
def flip_v(grid): return from_numpy(np.flipud(to_numpy(grid)))
def transpose(grid): return from_numpy(to_numpy(grid).T)


def translate(grid, dr: int, dc: int, fill: int = 0):
    h, w = shape(grid)
    out = np.full((h, w), fill, dtype=np.int64)
    arr = to_numpy(grid, fill)
    for r in range(h):
        for c in range(w):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = arr[r, c]
    return from_numpy(out)


def overlay(base, top, transparent: int = 0, r0: int = 0, c0: int = 0):
    out = to_numpy(base)
    t = to_numpy(top)
    for r in range(t.shape[0]):
        for c in range(t.shape[1]):
            if int(t[r, c]) != transparent and 0 <= r + r0 < out.shape[0] and 0 <= c + c0 < out.shape[1]:
                out[r + r0, c + c0] = t[r, c]
    return from_numpy(out)


def replace_color(grid, old: int, new: int):
    arr = to_numpy(grid)
    arr[arr == old] = new
    return from_numpy(arr)


def mask_color(grid, color: int):
    return from_numpy((to_numpy(grid) == color).astype(np.int64))


def grid_equal_with_missing_zero(a, b) -> bool:
    h = max(height(a), height(b))
    w = max(width(a), width(b))
    return all(get(a, r, c) == get(b, r, c) for r in range(h) for c in range(w))
