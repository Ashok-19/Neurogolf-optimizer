from __future__ import annotations

import onnx
import numpy as np
from onnx import numpy_helper

from ..graph_utils import clone_model, get_tensor_elem_type, set_initializer
from ..models import BuildResult, Mutation, TaskContext
from ..mutation_base import BaseMutationFamily, mutation_id


class ScalarLabelRewrite(BaseMutationFamily):
    name = "scalar_label_rewrite"

    def discover(self, task: TaskContext, model: onnx.ModelProto) -> list[Mutation]:
        muts = []
        arrays = {}
        for init in model.graph.initializer:
            try:
                arrays[init.name] = numpy_helper.to_array(init)
            except Exception:
                continue
        for idx, node in enumerate(model.graph.node):
            if node.op_type == "Equal" and len(node.input) == 2:
                for inp in node.input:
                    arr = arrays.get(inp)
                    if arr is None or arr.size == 0:
                        continue
                    if arr.min(initial=0) >= 0 and arr.max(initial=0) <= 255 and arr.dtype != np.uint8:
                        muts.append(
                            Mutation(
                                id=mutation_id(self.name, "equal_color_uint8", idx, inp),
                                family=self.name,
                                description=f"lower Equal color/label initializer {inp} to uint8",
                                priority=150.0 + arr.size / 1000.0,
                                estimated_savings=int(arr.size * max(1, arr.dtype.itemsize - 1)),
                                metadata={"kind": "equal_initializer_to_uint8", "node_index": idx, "initializer": inp, "inputs": list(node.input), "outputs": list(node.output)},
                            )
                        )
        return muts[: min(len(muts), task.config.beam_size)]

    def apply(self, task: TaskContext, model: onnx.ModelProto, mutation: Mutation) -> BuildResult:
        if mutation.metadata.get("kind") != "equal_initializer_to_uint8":
            return BuildResult(model=None, ok=False, error="unsupported scalar-label rewrite")
        m = clone_model(model)
        name = mutation.metadata["initializer"]
        for init in m.graph.initializer:
            if init.name == name:
                arr = numpy_helper.to_array(init)
                if arr.min(initial=0) < 0 or arr.max(initial=0) > 255:
                    return BuildResult(model=None, ok=False, error="initializer outside uint8 range")
                set_initializer(m, name, arr.astype(np.uint8))
                return self.build(task, m)
        return BuildResult(model=None, ok=False, error="initializer not found")
