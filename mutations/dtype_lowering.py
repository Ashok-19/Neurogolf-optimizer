from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from ..graph_utils import clone_model
from ..models import BuildResult, Mutation, TaskContext
from ..mutation_base import BaseMutationFamily, mutation_id


class DtypeLowering(BaseMutationFamily):
    name = "dtype_lowering"

    def discover(self, task: TaskContext, model: onnx.ModelProto) -> list[Mutation]:
        consumers: dict[str, set[str]] = {}
        for node in model.graph.node:
            for inp in node.input:
                consumers.setdefault(inp, set()).add(node.op_type)
        muts = []
        for init in model.graph.initializer:
            try:
                arr = numpy_helper.to_array(init)
            except Exception:
                continue
            ops = consumers.get(init.name, set())
            if arr.size == 0:
                continue
            if init.data_type == TensorProto.FLOAT and {"Conv", "Gemm", "MatMul", "Add", "Sub", "Mul", "Div", "Where"}.isdisjoint(ops):
                muts.append(self._mutation(init.name, "float32_to_float16", int(arr.size * 2), 220.0))
                if np.all(np.isfinite(arr)) and np.all(arr == np.floor(arr)) and arr.min(initial=0) >= 0 and arr.max(initial=0) <= 255:
                    muts.append(self._mutation(init.name, "float32_to_uint8", int(arr.size * 3), 218.0))
            if init.data_type == TensorProto.FLOAT16 and np.all(arr == np.floor(arr)) and arr.min(initial=0) >= 0 and arr.max(initial=0) <= 255:
                muts.append(self._mutation(init.name, "float16_to_uint8", int(arr.size), 216.0))
            if init.data_type == TensorProto.INT64 and arr.min(initial=0) >= np.iinfo(np.int32).min and arr.max(initial=0) <= np.iinfo(np.int32).max:
                if ops and ops <= {"Gather", "GatherND", "GatherElements"}:
                    muts.append(self._mutation(init.name, "int64_to_int32", int(arr.size * 4), 210.0))
            if init.data_type in (TensorProto.UINT8, TensorProto.INT64, TensorProto.INT32) and arr.size >= 16 and np.all((arr == 0) | (arr == 1)):
                if ops <= {"Where", "And", "Or", "Not", "Xor", "Equal"} or not ops:
                    muts.append(self._mutation(init.name, "zero_one_to_bool", int(arr.size), 205.0))
        for idx, node in enumerate(model.graph.node):
            if node.op_type != "Constant" or not node.output:
                continue
            for attr in node.attribute:
                if attr.name == "value" and attr.HasField("t"):
                    arr = numpy_helper.to_array(attr.t)
                    if attr.t.data_type == TensorProto.FLOAT and arr.size and np.all(arr == np.floor(arr)) and arr.min(initial=0) >= 0 and arr.max(initial=0) <= 255:
                        muts.append(
                            Mutation(
                                id=mutation_id(self.name, "constant_float_to_uint8", idx, node.output[0]),
                                family=self.name,
                                description=f"lower Constant {node.output[0]} float32 to uint8",
                                priority=200.0 + arr.size / 1000.0,
                                estimated_savings=int(arr.size * 3),
                                metadata={"kind": "constant_float32_to_uint8", "node_index": idx, "output": node.output[0]},
                            )
                        )
                    if attr.t.data_type == TensorProto.FLOAT and arr.size:
                        muts.append(
                            Mutation(
                                id=mutation_id(self.name, "constant_float_to_float16", idx, node.output[0]),
                                family=self.name,
                                description=f"lower Constant {node.output[0]} float32 to float16",
                                priority=198.0 + arr.size / 1000.0,
                                estimated_savings=int(arr.size * 2),
                                metadata={"kind": "constant_float32_to_float16", "node_index": idx, "output": node.output[0]},
                            )
                        )
        muts.sort(key=lambda m: (m.estimated_savings or 0, m.priority), reverse=True)
        return muts[: task.config.max_candidates_per_family]

    def _mutation(self, name: str, kind: str, savings: int, priority: float) -> Mutation:
        return Mutation(
            id=mutation_id(self.name, kind, name),
            family=self.name,
            description=f"lower initializer {name}: {kind}",
            priority=priority + savings / 1000.0,
            estimated_savings=savings,
            metadata={"initializer": name, "kind": kind},
        )

    def apply(self, task: TaskContext, model: onnx.ModelProto, mutation: Mutation) -> BuildResult:
        m = clone_model(model)
        name = mutation.metadata["initializer"]
        kind = mutation.metadata["kind"]
        for i, init in enumerate(m.graph.initializer):
            if init.name != name:
                continue
            arr = numpy_helper.to_array(init)
            if kind == "float32_to_float16":
                new = arr.astype(np.float16)
            elif kind == "float32_to_uint8":
                new = arr.astype(np.uint8)
            elif kind == "float16_to_uint8":
                new = arr.astype(np.uint8)
            elif kind == "int64_to_int32":
                new = arr.astype(np.int32)
            elif kind == "zero_one_to_bool":
                new = arr.astype(bool)
            else:
                return BuildResult(model=None, ok=False, error=f"unknown dtype lowering {kind}")
            m.graph.initializer.remove(init)
            m.graph.initializer.insert(i, numpy_helper.from_array(new, name=name))
            return self.build(task, m)
        if kind in {"constant_float32_to_uint8", "constant_float32_to_float16"}:
            idx = int(mutation.metadata["node_index"])
            if idx >= len(m.graph.node) or m.graph.node[idx].op_type != "Constant":
                return BuildResult(model=None, ok=False, error="constant node not found")
            node = m.graph.node[idx]
            for attr in node.attribute:
                if attr.name == "value" and attr.HasField("t"):
                    arr = numpy_helper.to_array(attr.t)
                    new = arr.astype(np.uint8 if kind.endswith("uint8") else np.float16)
                    attr.t.CopyFrom(numpy_helper.from_array(new, name=attr.t.name))
                    return self.build(task, m)
        return BuildResult(model=None, ok=False, error="initializer not found")
