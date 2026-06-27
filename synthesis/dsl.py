from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import grid as gu
from .objects import extract_objects, extract_color_objects


@dataclass(frozen=True)
class Step:
    op: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Program:
    name: str
    family: str
    params: dict[str, Any] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)

    def run_python(self, input_grid):
        cur = input_grid
        for step in self.steps:
            p = step.params
            if step.op == "Input":
                cur = input_grid
            elif step.op == "Recolor":
                mapping = {int(k): int(v) for k, v in p["map"].items()}
                cur = [[mapping.get(v, v) for v in row] for row in cur]
            elif step.op == "ReplaceColor":
                cur = gu.replace_color(cur, int(p["old"]), int(p["new"]))
            elif step.op == "Crop":
                cur = gu.crop(cur, tuple(p["bbox"]))
            elif step.op == "CropNonZeroBBox":
                cur = gu.crop(cur, gu.bounding_box(cur))
            elif step.op == "CropColorBBox":
                cur = gu.crop(cur, gu.bounding_box(cur, int(p["color"])))
            elif step.op == "SelectColor":
                cur = gu.mask_color(cur, int(p["color"]))
            elif step.op == "SelectNonZero":
                cur = [[1 if v != int(p.get("background", 0)) else 0 for v in row] for row in cur]
            elif step.op == "ConnectedComponents":
                cur = cur
            elif step.op == "LargestObject":
                objs = extract_objects(cur, connectivity=int(p.get("connectivity", 4)))
                obj = max(objs, key=lambda o: o.area) if objs else None
                cur = gu.crop(cur, obj.bbox if obj else None)
            elif step.op == "SmallestObject":
                objs = extract_objects(cur, connectivity=int(p.get("connectivity", 4)))
                obj = min(objs, key=lambda o: o.area) if objs else None
                cur = gu.crop(cur, obj.bbox if obj else None)
            elif step.op == "ObjectByColor":
                cur = gu.crop(cur, gu.bounding_box(cur, int(p["color"])))
            elif step.op == "BoundingBox":
                cur = gu.crop(cur, gu.bounding_box(cur, p.get("color")))
            elif step.op == "CropBBox":
                cur = gu.crop(cur, gu.bounding_box(cur, p.get("color")))
            elif step.op == "MakeCanvas":
                cur = [[int(p.get("fill", 0)) for _ in range(int(p["width"]))] for _ in range(int(p["height"]))]
            elif step.op == "DrawRectangle":
                arr = gu.to_numpy(cur)
                bbox = gu.bounding_box(cur, p.get("target_color"))
                if bbox is not None:
                    r0, c0, r1, c1 = bbox
                    color = int(p.get("draw_color", p.get("target_color", 1)))
                    arr[r0:r1, c0] = color
                    arr[r0:r1, c1 - 1] = color
                    arr[r0, c0:c1] = color
                    arr[r1 - 1, c0:c1] = color
                cur = gu.from_numpy(arr)
            elif step.op == "FillRectangle":
                arr = gu.to_numpy(cur)
                bbox = gu.bounding_box(cur, p.get("target_color"))
                if bbox is not None:
                    r0, c0, r1, c1 = bbox
                    arr[r0:r1, c0:c1] = int(p.get("fill_color", p.get("target_color", 1)))
                cur = gu.from_numpy(arr)
            elif step.op == "DrawLine":
                cur = _draw_line(cur, p)
            elif step.op == "DrawRay":
                cur = _draw_ray(cur, p)
            elif step.op == "MaskAnd":
                cur = gu.from_numpy((gu.to_numpy(cur) & gu.to_numpy(p["mask"])).astype(int))
            elif step.op == "MaskOr":
                cur = gu.from_numpy((gu.to_numpy(cur) | gu.to_numpy(p["mask"])).astype(int))
            elif step.op == "MaskNot":
                cur = gu.from_numpy((gu.to_numpy(cur) == 0).astype(int))
            elif step.op == "Overlay":
                cur = gu.overlay(cur, p["top"], transparent=int(p.get("transparent", 0)))
            elif step.op == "PasteObject":
                cur = gu.overlay(cur, p["object"], transparent=int(p.get("transparent", 0)), r0=int(p.get("r", 0)), c0=int(p.get("c", 0)))
            elif step.op == "Tile":
                arr = gu.to_numpy(cur)
                reps = (int(p.get("height", 30)) // max(1, arr.shape[0]) + 1, int(p.get("width", 30)) // max(1, arr.shape[1]) + 1)
                cur = gu.from_numpy(np.tile(arr, reps)[: int(p.get("height", 30)), : int(p.get("width", 30))])
            elif step.op == "Scale":
                arr = gu.to_numpy(cur)
                k = int(p.get("factor", 2))
                cur = gu.from_numpy(np.repeat(np.repeat(arr, k, axis=0), k, axis=1))
            elif step.op == "Gravity":
                cur = _gravity(cur, p.get("direction", "down"))
            elif step.op == "ExtractBorder":
                arr = gu.to_numpy(cur)
                out = np.zeros_like(arr)
                if arr.size:
                    out[0, :] = arr[0, :]
                    out[-1, :] = arr[-1, :]
                    out[:, 0] = arr[:, 0]
                    out[:, -1] = arr[:, -1]
                cur = gu.from_numpy(out)
            elif step.op == "RemoveBorder":
                h, w = gu.shape(cur)
                cur = gu.crop(cur, (1, 1, max(1, h - 1), max(1, w - 1)))
            elif step.op == "Rotate":
                cur = {90: gu.rotate90, 180: gu.rotate180, 270: gu.rotate270}[int(p["deg"])](cur)
            elif step.op == "Flip":
                cur = gu.flip_h(cur) if p["axis"] == "h" else gu.flip_v(cur)
            elif step.op == "PadTo30":
                cur = gu.pad_to_30(cur, int(p.get("fill", 0)))
            elif step.op == "Constant":
                cur = p["grid"]
            else:
                raise ValueError(f"unsupported DSL op {step.op}")
        return cur

    def describe(self) -> str:
        return f"{self.family}:{self.name} " + " -> ".join(s.op for s in self.steps)

    def complexity_score(self) -> int:
        return len(self.steps) + sum(len(str(s.params)) // 10 for s in self.steps)

    def to_dict(self) -> dict:
        return {"name": self.name, "family": self.family, "params": self.params, "steps": [{"op": s.op, "params": s.params} for s in self.steps]}

    @classmethod
    def from_dict(cls, data: dict) -> "Program":
        return cls(data["name"], data["family"], data.get("params", {}), [Step(s["op"], s.get("params", {})) for s in data.get("steps", [])])


def _draw_line(grid, p):
    arr = gu.to_numpy(grid)
    color = int(p.get("color", 1))
    if "row" in p:
        arr[int(p["row"]), :] = color
    elif "col" in p:
        arr[:, int(p["col"])] = color
    elif all(k in p for k in ("r0", "c0", "r1", "c1")):
        r0, c0, r1, c1 = map(int, (p["r0"], p["c0"], p["r1"], p["c1"]))
        steps = max(abs(r1 - r0), abs(c1 - c0)) + 1
        for i in range(steps):
            r = round(r0 + (r1 - r0) * i / max(1, steps - 1))
            c = round(c0 + (c1 - c0) * i / max(1, steps - 1))
            if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
                arr[r, c] = color
    return gu.from_numpy(arr)


def _draw_ray(grid, p):
    arr = gu.to_numpy(grid)
    color = int(p.get("color", 1))
    r, c = int(p.get("r", 0)), int(p.get("c", 0))
    dr, dc = int(p.get("dr", 0)), int(p.get("dc", 1))
    while 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
        arr[r, c] = color
        r += dr
        c += dc
    return gu.from_numpy(arr)


def _gravity(grid, direction):
    arr = gu.to_numpy(grid)
    out = np.zeros_like(arr)
    if direction in ("down", "up"):
        for c in range(arr.shape[1]):
            vals = [v for v in arr[:, c] if v != 0]
            if direction == "down":
                out[arr.shape[0] - len(vals) :, c] = vals
            else:
                out[: len(vals), c] = vals
    else:
        for r in range(arr.shape[0]):
            vals = [v for v in arr[r, :] if v != 0]
            if direction == "right":
                out[r, arr.shape[1] - len(vals) :] = vals
            else:
                out[r, : len(vals)] = vals
    return gu.from_numpy(out)
