from __future__ import annotations

from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

from ..graph_utils import clone_model, get_consumers, get_value_info_shapes
from ..models import BuildResult, Mutation, TaskContext
from ..mutation_base import BaseMutationFamily, mutation_id


BROADCAST_SAFE_OPS = {
    "Add",
    "And",
    "Div",
    "Equal",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
    "Mul",
    "Or",
    "Sub",
    "Where",
    "Xor",
}


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _static_shape(shape: list[int | None] | None) -> tuple[int, ...] | None:
    if not shape or any(dim is None for dim in shape):
        return None
    return tuple(int(dim) for dim in shape)


def _broadcast_shape(shapes: list[tuple[int, ...]]) -> tuple[int, ...] | None:
    try:
        return tuple(np.broadcast_shapes(*shapes))
    except ValueError:
        return None


def can_scalarize_initializer_for_node(
    node: onnx.NodeProto,
    initializer: str,
    original_shape: list[int | None],
    shapes: dict[str, list[int | None]],
) -> bool:
    if node.op_type not in BROADCAST_SAFE_OPS:
        return False
    if node.op_type == "Where" and node.input and node.input[0] == initializer:
        return False
    old_inputs: list[tuple[int, ...]] = []
    new_inputs: list[tuple[int, ...]] = []
    for value in node.input:
        if not value:
            continue
        if value == initializer:
            old_shape = _static_shape(original_shape)
            if old_shape is None:
                return False
            old_inputs.append(old_shape)
            new_inputs.append(())
            continue
        shape = _static_shape(shapes.get(value))
        if shape is None:
            return False
        old_inputs.append(shape)
        new_inputs.append(shape)
    old_output_shape = _broadcast_shape(old_inputs)
    new_output_shape = _broadcast_shape(new_inputs)
    if old_output_shape is None or new_output_shape is None:
        return False
    expected = _static_shape(shapes.get(node.output[0] if node.output else ""))
    if expected is None:
        expected = old_output_shape
    return new_output_shape == expected


class BroadcastScalarize(BaseMutationFamily):
    name = "broadcast_scalarize"

    def discover(self, task: TaskContext, model: onnx.ModelProto) -> list[Mutation]:
        consumers = get_consumers(model)
        shapes = get_value_info_shapes(model)
        muts: list[Mutation] = []
        for init in model.graph.initializer:
            try:
                arr = numpy_helper.to_array(init)
            except Exception:
                continue
            if arr.size < 16:
                continue
            flat = arr.reshape(-1)
            if flat.size == 0 or not np.all(flat == flat[0]):
                continue
            init_consumers = consumers.get(init.name, [])
            if not init_consumers:
                continue
            if not all(node.op_type in BROADCAST_SAFE_OPS for node in init_consumers):
                continue
            if not all(can_scalarize_initializer_for_node(node, init.name, list(arr.shape), shapes) for node in init_consumers):
                continue
            savings = int(arr.size - 1)
            dtype_name = str(arr.dtype)
            op_names = sorted({node.op_type for node in init_consumers})
            muts.append(
                Mutation(
                    id=mutation_id(self.name, init.name, dtype_name, tuple(arr.shape), _json_scalar(flat[0])),
                    family=self.name,
                    description=f"scalarize uniform broadcast initializer {init.name} for {','.join(op_names)}",
                    priority=260.0 + savings / 1000.0,
                    estimated_savings=savings,
                    metadata={
                        "initializer": init.name,
                        "dtype": dtype_name,
                        "shape": list(arr.shape),
                        "value": _json_scalar(flat[0]),
                        "consumers": [{"op": node.op_type, "name": node.name, "outputs": list(node.output)} for node in init_consumers],
                    },
                )
            )
        muts.sort(key=lambda m: (m.estimated_savings or 0, m.priority), reverse=True)
        return muts[: task.config.max_candidates_per_family]

    def apply(self, task: TaskContext, model: onnx.ModelProto, mutation: Mutation) -> BuildResult:
        m = clone_model(model)
        name = mutation.metadata["initializer"]
        for idx, init in enumerate(list(m.graph.initializer)):
            if init.name != name:
                continue
            arr = numpy_helper.to_array(init)
            if arr.size == 0:
                return BuildResult(model=None, ok=False, error="initializer is empty")
            flat = arr.reshape(-1)
            if not np.all(flat == flat[0]):
                return BuildResult(model=None, ok=False, error="initializer is no longer uniform")
            scalar = np.array(_json_scalar(flat[0]), dtype=arr.dtype)
            m.graph.initializer.remove(init)
            m.graph.initializer.insert(idx, numpy_helper.from_array(scalar, name=name))
            return self.build(task, m)
        return BuildResult(model=None, ok=False, error="initializer not found")
