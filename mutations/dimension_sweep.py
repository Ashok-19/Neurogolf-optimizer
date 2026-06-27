from __future__ import annotations

import numpy as np
import onnx
from onnx import numpy_helper

from ..graph_utils import clone_model, initializer_arrays, replace_initializer
from ..models import BuildResult, Mutation, TaskContext
from ..mutation_base import BaseMutationFamily, mutation_id


def _is_arange(arr: np.ndarray) -> bool:
    flat = np.asarray(arr).reshape(-1)
    return flat.size >= 4 and flat.size <= 30 and np.array_equal(flat, np.arange(flat.size, dtype=flat.dtype))


def _axis_from_name(name: str) -> str | None:
    low = name.lower()
    if "row" in low or low.startswith("r"):
        return "row"
    if "col" in low or low.startswith("c"):
        return "col"
    return None


def _axis_from_shape(arr: np.ndarray) -> str | None:
    if arr.ndim == 4:
        if arr.shape[2] > 1 and arr.shape[3] == 1:
            return "row"
        if arr.shape[2] == 1 and arr.shape[3] > 1:
            return "col"
    return None


class DimensionSweep(BaseMutationFamily):
    name = "dimension_sweep"

    def discover(self, task: TaskContext, model: onnx.ModelProto) -> list[Mutation]:
        arrays = initializer_arrays(model)
        out: list[Mutation] = []
        for init_name, arr in arrays.items():
            if not np.issubdtype(arr.dtype, np.integer) or not _is_arange(arr):
                continue
            old_len = int(arr.size)
            if old_len <= 4:
                continue
            for node in model.graph.node:
                for input_index, inp in enumerate(node.input):
                    if inp != init_name:
                        continue
                    axis = _axis_from_name(node.name) or _axis_from_name(node.output[0] if node.output else "")
                    if axis is None:
                        continue
                    for new_len in range(old_len - 1, max(3, old_len - 4), -1):
                        meta = {
                            "source_idx": init_name,
                            "node_name": node.name,
                            "node_output": list(node.output),
                            "input_index": input_index,
                            "axis": axis,
                            "old_len": old_len,
                            "new_len": new_len,
                        }
                        out.append(
                            Mutation(
                                id=mutation_id(self.name, meta),
                                family=self.name,
                                description=f"reduce {axis} work dimension {old_len}->{new_len} for {node.name}",
                                priority=420.0 + (old_len - new_len),
                                estimated_savings=max(1, old_len - new_len) * old_len,
                                metadata=meta,
                            )
                        )
        return out[:128]

    def apply(self, task: TaskContext, model: onnx.ModelProto, mutation: Mutation) -> BuildResult:
        meta = mutation.metadata
        axis = str(meta["axis"])
        old_len = int(meta["old_len"])
        new_len = int(meta["new_len"])
        source_idx = str(meta["source_idx"])
        changed = False
        out = clone_model(model)
        new_idx = f"{source_idx}_{axis}_{new_len}"
        out.graph.initializer.append(numpy_helper.from_array(np.arange(new_len, dtype=np.int32), name=new_idx))
        for node in out.graph.node:
            if node.name == meta["node_name"] and int(meta["input_index"]) < len(node.input):
                node.input[int(meta["input_index"])] = new_idx
                changed = True
        for init in list(out.graph.initializer):
            if init.name == new_idx:
                continue
            try:
                arr = numpy_helper.to_array(init)
            except Exception:
                continue
            if not np.issubdtype(arr.dtype, np.integer):
                continue
            init_axis = _axis_from_name(init.name) or _axis_from_shape(arr)
            if init_axis != axis:
                continue
            if arr.ndim == 4 and axis == "row" and arr.shape[2] == old_len and arr.shape[3] == 1 and _is_arange(arr):
                replace_initializer(out, init.name, np.arange(new_len, dtype=arr.dtype).reshape(1, 1, new_len, 1))
                changed = True
            elif arr.ndim == 4 and axis == "col" and arr.shape[2] == 1 and arr.shape[3] == old_len and _is_arange(arr):
                replace_initializer(out, init.name, np.arange(new_len, dtype=arr.dtype).reshape(1, 1, 1, new_len))
                changed = True
        for init in list(out.graph.initializer):
            if init.name != "pad_to_30_i64":
                continue
            pads = numpy_helper.to_array(init).astype(np.int64)
            if pads.shape == (8,):
                pads = pads.copy()
                if axis == "row":
                    pads[6] = max(0, 30 - new_len)
                elif axis == "col":
                    pads[7] = max(0, 30 - new_len)
                replace_initializer(out, init.name, pads)
                changed = True
        del out.graph.value_info[:]
        return self.build(task, out, changed=changed)
