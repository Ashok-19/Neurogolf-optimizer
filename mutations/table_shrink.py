from __future__ import annotations

from collections import defaultdict

import numpy as np
import onnx
from onnx import numpy_helper

from ..graph_utils import clone_model, get_consumers, set_initializer
from ..models import BuildResult, Mutation, TaskContext
from ..mutation_base import BaseMutationFamily, mutation_id
from .gather_shrink import GatherShrink, _axis, _normalize_indices


class TableShrink(BaseMutationFamily):
    name = "table_shrink"

    def discover(self, task: TaskContext, model: onnx.ModelProto) -> list[Mutation]:
        arrays = {}
        for init in model.graph.initializer:
            try:
                arrays[init.name] = numpy_helper.to_array(init)
            except Exception:
                continue
        muts: list[Mutation] = []
        muts.extend(GatherShrink().discover(task, model))
        muts.extend(self._discover_row_dedup(task, model, arrays))
        muts.extend(self._discover_slice_delete(task, model, arrays))
        muts.sort(key=lambda m: (m.priority, m.estimated_savings or 0), reverse=True)
        return muts[: task.config.max_candidates_per_family]

    def _discover_row_dedup(self, task, model, arrays) -> list[Mutation]:
        consumers = get_consumers(model)
        muts = []
        for name, arr in arrays.items():
            if arr.ndim < 2 or arr.shape[0] < 4 or arr.size < 128:
                continue
            table_consumers = consumers.get(name, [])
            if not table_consumers or any(n.op_type != "Gather" for n in table_consumers):
                continue
            index_names = []
            ok = True
            for node in table_consumers:
                if _axis(node) != 0 or len(node.input) < 2 or node.input[1] not in arrays:
                    ok = False
                    break
                idx_arr = arrays[node.input[1]]
                if _normalize_indices(idx_arr, arr.shape[0]) is None:
                    ok = False
                    break
                index_names.append(node.input[1])
            if not ok:
                continue
            rows = arr.reshape(arr.shape[0], -1)
            unique, inverse = np.unique(rows, axis=0, return_inverse=True)
            if unique.shape[0] >= rows.shape[0]:
                continue
            savings = int((rows.shape[0] - unique.shape[0]) * rows.shape[1])
            muts.append(
                Mutation(
                    id=mutation_id(self.name, "row_dedup", name, unique.shape[0], rows.shape[0], tuple(sorted(set(index_names)))),
                    family=self.name,
                    description=f"deduplicate table {name} rows {rows.shape[0]} -> {unique.shape[0]}",
                    priority=310.0 + savings / 1000.0,
                    estimated_savings=savings,
                    metadata={"kind": "row_dedup", "initializer": name, "index_initializers": sorted(set(index_names))},
                )
            )
        return muts

    def _discover_slice_delete(self, task, model, arrays) -> list[Mutation]:
        muts = []
        for name, arr in arrays.items():
            if arr.ndim < 2 or arr.size < 512:
                continue
            for axis, dim in enumerate(arr.shape):
                if dim <= 1 or dim > 64:
                    continue
                slice_size = int(arr.size / dim)
                if slice_size <= 0:
                    continue
                # Probe tail and fallback-like slices first to keep candidate count bounded.
                for k in sorted({dim - 1, 0, dim // 2}):
                    muts.append(
                        Mutation(
                            id=mutation_id(self.name, "delete_slice", name, axis, k),
                            family=self.name,
                            description=f"probe deleting slice {k} on axis {axis} from {name}",
                            priority=120.0 + slice_size / 1000.0,
                            estimated_savings=slice_size,
                            metadata={"kind": "delete_slice_probe", "initializer": name, "axis": axis, "slice": k},
                        )
                    )
        return muts

    def apply(self, task: TaskContext, model: onnx.ModelProto, mutation: Mutation) -> BuildResult:
        if mutation.family == "gather_shrink" or mutation.metadata.get("kind", "").startswith("gather"):
            return GatherShrink().apply(task, model, mutation)
        kind = mutation.metadata.get("kind")
        if kind == "row_dedup":
            return self._apply_row_dedup(task, model, mutation)
        if kind == "delete_slice_probe":
            return self._apply_delete_slice(task, model, mutation)
        return BuildResult(model=None, ok=False, error=f"unsupported table shrink {kind}")

    def _apply_row_dedup(self, task, model, mutation):
        m = clone_model(model)
        name = mutation.metadata["initializer"]
        arrays = {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}
        arr = arrays[name]
        rows = arr.reshape(arr.shape[0], -1)
        unique, inverse = np.unique(rows, axis=0, return_inverse=True)
        new_arr = unique.reshape((unique.shape[0],) + arr.shape[1:]).astype(arr.dtype, copy=False)
        set_initializer(m, name, new_arr)
        for idx_name in mutation.metadata["index_initializers"]:
            idx = arrays[idx_name]
            norm = _normalize_indices(idx, arr.shape[0])
            if norm is None:
                return BuildResult(model=None, ok=False, error=f"cannot normalize {idx_name}")
            remapped = inverse[norm].astype(idx.dtype)
            set_initializer(m, idx_name, remapped)
        return self.build(task, m)

    def _apply_delete_slice(self, task, model, mutation):
        m = clone_model(model)
        name = mutation.metadata["initializer"]
        axis = int(mutation.metadata["axis"])
        k = int(mutation.metadata["slice"])
        arrays = {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}
        arr = arrays.get(name)
        if arr is None or axis >= arr.ndim or arr.shape[axis] <= 1:
            return BuildResult(model=None, ok=False, error="slice target invalid")
        new_arr = np.delete(arr, k, axis=axis)
        set_initializer(m, name, new_arr)
        # Only remap direct Gather index initializers when this table is gathered along the same axis.
        for node in get_consumers(m).get(name, []):
            if node.op_type == "Gather" and _axis(node) == axis and len(node.input) >= 2 and node.input[1] in arrays:
                idx = arrays[node.input[1]]
                norm = _normalize_indices(idx, arr.shape[axis])
                if norm is None or np.any(norm == k):
                    return BuildResult(model=None, ok=False, error="deleted slice is referenced")
                remapped = norm.copy()
                remapped[remapped > k] -= 1
                set_initializer(m, node.input[1], remapped.astype(idx.dtype))
            else:
                return BuildResult(model=None, ok=False, error="dynamic or unsupported table consumer")
        return self.build(task, m)
