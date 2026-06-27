from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from ..graph_utils import clone_model, set_initializer
from ..models import BuildResult, Mutation, TaskContext
from ..mutation_base import BaseMutationFamily, mutation_id


def _axis(node: onnx.NodeProto, default: int = 0) -> int:
    for attr in node.attribute:
        if attr.name == "axis":
            return int(attr.i)
    return default


def _normalize_indices(indices: np.ndarray, dim: int) -> np.ndarray | None:
    if not np.issubdtype(indices.dtype, np.integer):
        return None
    out = indices.astype(np.int64, copy=True)
    out[out < 0] += dim
    if out.size and (out.min() < 0 or out.max() >= dim):
        return None
    return out


class GatherShrink(BaseMutationFamily):
    name = "gather_shrink"

    def discover(self, task: TaskContext, model: onnx.ModelProto) -> list[Mutation]:
        init_arrays = {}
        for init in model.graph.initializer:
            try:
                init_arrays[init.name] = numpy_helper.to_array(init)
            except Exception:
                continue
        muts: list[Mutation] = []
        for idx, node in enumerate(model.graph.node):
            if len(node.input) < 2:
                continue
            if node.op_type == "Gather":
                muts.extend(self._discover_gather(task, idx, node, init_arrays))
            elif node.op_type == "GatherND":
                muts.extend(self._discover_gathernd(task, idx, node, init_arrays))
            elif node.op_type == "GatherElements":
                muts.extend(self._discover_gather_elements(task, idx, node, init_arrays))
        muts.sort(key=lambda m: (m.priority, m.estimated_savings or 0), reverse=True)
        return muts[: task.config.max_candidates_per_family]

    def _discover_gather(self, task, idx, node, arrays) -> list[Mutation]:
        data_name, idx_name = node.input[0], node.input[1]
        data = arrays.get(data_name)
        indices = arrays.get(idx_name)
        if data is None or indices is None or data.ndim == 0 or indices.size == 0:
            return []
        axis = _axis(node)
        if axis < 0:
            axis += data.ndim
        if axis < 0 or axis >= data.ndim:
            return []
        norm = _normalize_indices(indices, data.shape[axis])
        if norm is None:
            return []
        used = sorted({int(x) for x in norm.reshape(-1)})
        if not used or len(used) >= data.shape[axis]:
            return []
        slice_size = int(np.prod([d for i, d in enumerate(data.shape) if i != axis]))
        savings = int((data.shape[axis] - len(used)) * slice_size)
        return [
            Mutation(
                id=mutation_id(self.name, "gather", idx, data_name, idx_name, axis, tuple(used)),
                family=self.name,
                description=f"shrink Gather table {data_name} axis {axis} to {len(used)}/{data.shape[axis]} slices",
                priority=340.0 + savings / 1000.0,
                estimated_savings=savings,
                metadata={"kind": "gather_axis", "node_index": idx, "data": data_name, "indices": idx_name, "axis": axis, "used": used},
            )
        ]

    def _discover_gathernd(self, task, idx, node, arrays) -> list[Mutation]:
        data_name, idx_name = node.input[0], node.input[1]
        data = arrays.get(data_name)
        indices = arrays.get(idx_name)
        if data is None or indices is None or data.ndim == 0 or indices.size == 0 or indices.shape[-1] < 1:
            return []
        first = _normalize_indices(indices[..., 0], data.shape[0])
        if first is None:
            return []
        used = sorted({int(x) for x in first.reshape(-1)})
        if not used or len(used) >= data.shape[0]:
            return []
        savings = int((data.shape[0] - len(used)) * np.prod(data.shape[1:]))
        return [
            Mutation(
                id=mutation_id(self.name, "gathernd", idx, data_name, idx_name, tuple(used)),
                family=self.name,
                description=f"shrink GatherND table {data_name} first axis to {len(used)}/{data.shape[0]} slices",
                priority=325.0 + savings / 1000.0,
                estimated_savings=savings,
                metadata={"kind": "gathernd_first_axis", "node_index": idx, "data": data_name, "indices": idx_name, "used": used},
            )
        ]

    def _discover_gather_elements(self, task, idx, node, arrays) -> list[Mutation]:
        data_name, idx_name = node.input[0], node.input[1]
        data = arrays.get(data_name)
        indices = arrays.get(idx_name)
        if data is None or indices is None or data.ndim == 0:
            return []
        axis = _axis(node)
        if axis < 0:
            axis += data.ndim
        if axis < 0 or axis >= data.ndim:
            return []
        norm = _normalize_indices(indices, data.shape[axis])
        if norm is None:
            return []
        used = sorted({int(x) for x in norm.reshape(-1)})
        if not used or len(used) >= data.shape[axis]:
            return []
        savings = int((data.shape[axis] - len(used)) * np.prod([d for i, d in enumerate(data.shape) if i != axis]))
        return [
            Mutation(
                id=mutation_id(self.name, "gatherelements", idx, data_name, idx_name, axis, tuple(used)),
                family=self.name,
                description=f"shrink GatherElements data {data_name} axis {axis}",
                priority=300.0 + savings / 1000.0,
                estimated_savings=savings,
                metadata={"kind": "gather_elements_axis", "node_index": idx, "data": data_name, "indices": idx_name, "axis": axis, "used": used},
            )
        ]

    def apply(self, task: TaskContext, model: onnx.ModelProto, mutation: Mutation) -> BuildResult:
        m = clone_model(model)
        kind = mutation.metadata["kind"]
        if kind in {"gather_axis", "gather_elements_axis"}:
            return self._apply_axis(task, m, mutation)
        if kind == "gathernd_first_axis":
            return self._apply_gathernd(task, m, mutation)
        if kind == "int64_indices_to_int32":
            return self._apply_index_dtype(task, m, mutation)
        return BuildResult(model=None, ok=False, error=f"unsupported gather shrink {kind}")

    def _apply_axis(self, task, m, mutation):
        data_name = mutation.metadata["data"]
        idx_name = mutation.metadata["indices"]
        axis = int(mutation.metadata["axis"])
        used = list(mutation.metadata["used"])
        remap = {old: new for new, old in enumerate(used)}
        for init in list(m.graph.initializer):
            if init.name == data_name:
                arr = numpy_helper.to_array(init)
                new_arr = np.take(arr, used, axis=axis)
                set_initializer(m, init.name, new_arr)
            elif init.name == idx_name:
                arr = numpy_helper.to_array(init)
                dim = max(used) + 1
                norm = _normalize_indices(arr, dim)
                if norm is None:
                    norm = arr.astype(np.int64)
                    norm[norm < 0] += dim
                new_idx = np.vectorize(lambda x: remap[int(x)])(norm).astype(arr.dtype)
                set_initializer(m, init.name, new_idx)
        return self.build(task, m)

    def _apply_gathernd(self, task, m, mutation):
        data_name = mutation.metadata["data"]
        idx_name = mutation.metadata["indices"]
        used = list(mutation.metadata["used"])
        remap = {old: new for new, old in enumerate(used)}
        for init in list(m.graph.initializer):
            if init.name == data_name:
                arr = numpy_helper.to_array(init)
                set_initializer(m, init.name, arr[used])
            elif init.name == idx_name:
                arr = numpy_helper.to_array(init).copy()
                first = arr[..., 0].astype(np.int64)
                first[first < 0] += max(used) + 1
                arr[..., 0] = np.vectorize(lambda x: remap[int(x)])(first).astype(arr.dtype)
                set_initializer(m, init.name, arr)
        return self.build(task, m)

    def _apply_index_dtype(self, task, m, mutation):
        for init in list(m.graph.initializer):
            if init.name == mutation.metadata["indices"]:
                arr = numpy_helper.to_array(init)
                if arr.min(initial=0) < np.iinfo(np.int32).min or arr.max(initial=0) > np.iinfo(np.int32).max:
                    return BuildResult(model=None, ok=False, error="indices outside int32 range")
                set_initializer(m, init.name, arr.astype(np.int32))
                return self.build(task, m)
        return BuildResult(model=None, ok=False, error="indices initializer not found")
