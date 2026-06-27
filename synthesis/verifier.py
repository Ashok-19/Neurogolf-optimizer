from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import grid_equal_with_missing_zero, shape, colors
from .onnx_templates import onehot_tensor_from_grid
from .task_data import GridPair


@dataclass
class Failure:
    index: int
    source: str
    kind: str
    expected_shape: tuple[int, int]
    predicted_shape: tuple[int, int]


def classify_failure(pred, expected) -> str:
    if shape(pred) != shape(expected):
        return "shape_mismatch"
    pc, ec = colors(pred), colors(expected)
    if pc != ec:
        return "wrong_color_set"
    return "wrong_cells"


def _official_tensor_equal(pred, expected) -> bool:
    return np.array_equal(onehot_tensor_from_grid(pred), onehot_tensor_from_grid(expected))


def verify_program(program, pairs: list[GridPair], mode: str = "official_tensor") -> tuple[int, list[Failure]]:
    if mode not in {"official_tensor", "grid_relaxed"}:
        raise ValueError(f"unsupported verification mode {mode}")
    right = 0
    failures: list[Failure] = []
    for i, pair in enumerate(pairs):
        try:
            pred = program.run_python(pair.input)
        except Exception:
            pred = []
        if mode == "official_tensor":
            matched = _official_tensor_equal(pred, pair.output)
        else:
            matched = grid_equal_with_missing_zero(pred, pair.output)
        if matched:
            right += 1
        else:
            failures.append(Failure(i, pair.source, classify_failure(pred, pair.output), shape(pair.output), shape(pred)))
    return right, failures


def repair_program(program, failures: list[Failure]):
    # Conservative repair hook: only annotate failure diagnostics for now. Program-search uses this for ranking.
    return []
