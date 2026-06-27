from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FailureClass:
    kind: str
    rows: list[int]
    cols: list[int]
    channels: list[int]
    expected_shape: list[int] | None = None
    predicted_tensor_shape: list[int] | None = None
    source: str | None = None
    index: int | None = None
    missing_active_channels: int | None = None
    extra_active_channels: int | None = None
    no_color_cells: int | None = None
    reason: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nested_diff(failure: dict[str, Any]) -> dict[str, Any]:
    official = failure.get("official_public_arc")
    if isinstance(official, dict):
        merged = dict(official)
        merged.setdefault("source", failure.get("source"))
        merged.setdefault("index", failure.get("index"))
        return merged
    return failure


def _shape(diff: dict[str, Any]) -> list[int] | None:
    shape = diff.get("expected_grid_shape") or diff.get("expected_shape")
    if isinstance(shape, (list, tuple)) and len(shape) == 2:
        return [int(shape[0]), int(shape[1])]
    return None


def _missing_coords(diff: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for coord in diff.get("first_failing_coordinates") or []:
        reasons = set(coord.get("reason") or [])
        expected = [int(ch) for ch in coord.get("expected_tensor_active_channels") or []]
        predicted = [int(ch) for ch in coord.get("predicted_tensor_active_channels") or []]
        if not expected or predicted:
            continue
        if not ({"missing_active_channels", "no_color"} & reasons):
            continue
        row = int(coord.get("row", -1))
        col = int(coord.get("col", -1))
        if row >= 0 and col >= 0:
            out.append({"row": row, "col": col, "channels": expected})
    return out


def _has_extra_black_outside(diff: dict[str, Any]) -> bool:
    if int(diff.get("extra_active_channel_0_outside_expected_output_shape") or 0) > 0:
        return True
    for coord in diff.get("first_failing_coordinates") or []:
        if "extra_active_channel_0_outside_expected_output_shape" in set(coord.get("reason") or []):
            return True
    return False


def classify_failure(failure: dict[str, Any]) -> FailureClass | None:
    diff = _nested_diff(failure)
    shape = _shape(diff)
    source = diff.get("source") if diff.get("source") is not None else failure.get("source")
    index = diff.get("index") if diff.get("index") is not None else failure.get("index")
    predicted_shape = diff.get("predicted_tensor_shape")
    if _has_extra_black_outside(diff):
        return FailureClass(
            kind="extra_active_black_outside_shape",
            rows=[],
            cols=[],
            channels=[],
            expected_shape=shape,
            predicted_tensor_shape=list(predicted_shape) if isinstance(predicted_shape, list) else None,
            source=source,
            index=index,
            missing_active_channels=diff.get("missing_active_channels"),
            extra_active_channels=diff.get("extra_active_channels"),
            no_color_cells=diff.get("no_color_cells"),
        )

    cleaned = _missing_coords(diff)
    if not cleaned:
        if int(diff.get("extra_active_channels") or 0) > 0 and int(diff.get("missing_active_channels") or 0) > 0:
            kind = "wrong_color_region"
        else:
            return None
        return FailureClass(
            kind=kind,
            rows=[],
            cols=[],
            channels=[],
            expected_shape=shape,
            predicted_tensor_shape=list(predicted_shape) if isinstance(predicted_shape, list) else None,
            source=source,
            index=index,
            missing_active_channels=diff.get("missing_active_channels"),
            extra_active_channels=diff.get("extra_active_channels"),
            no_color_cells=diff.get("no_color_cells"),
        )

    rows = sorted({int(item["row"]) for item in cleaned})
    cols = sorted({int(item["col"]) for item in cleaned})
    channels = sorted({int(ch) for item in cleaned for ch in item["channels"]})
    h = shape[0] if shape else None
    w = shape[1] if shape else None
    kind = "missing_object_pixels"
    if len(cols) == 1 and len(rows) >= 2:
        kind = "missing_last_column" if w is not None and cols[0] == w - 1 else "missing_column"
    elif len(rows) == 1 and len(cols) >= 2:
        kind = "missing_bottom_row" if h is not None and rows[0] == h - 1 else "missing_row"
    elif h is not None and w is not None and ((h - 1 in rows) or (w - 1 in cols)):
        kind = "missing_border"

    return FailureClass(
        kind=kind,
        rows=rows,
        cols=cols,
        channels=channels,
        expected_shape=shape,
        predicted_tensor_shape=list(predicted_shape) if isinstance(predicted_shape, list) else None,
        source=source,
        index=index,
        missing_active_channels=diff.get("missing_active_channels"),
        extra_active_channels=diff.get("extra_active_channels"),
        no_color_cells=diff.get("no_color_cells"),
    )


def classify_diagnostics(diagnostics: dict[str, Any]) -> list[FailureClass]:
    out: list[FailureClass] = []
    seen: set[tuple[Any, ...]] = set()
    for failure in diagnostics.get("failures", []) or []:
        classified = classify_failure(failure)
        if classified is None:
            continue
        key = (
            classified.kind,
            tuple(classified.rows),
            tuple(classified.cols),
            tuple(classified.channels),
            tuple(classified.expected_shape or []),
            classified.source,
            classified.index,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(classified)
    return out
