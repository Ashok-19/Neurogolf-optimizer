from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..graph_utils import clone_model, initializer_signature
from ..models import BuildResult, Mutation, TaskContext
from ..mutation_base import BaseMutationFamily, mutation_id


class ConcatPadRewrite(BaseMutationFamily):
    name = "concat_pad"

    def discover(self, task: TaskContext, model: onnx.ModelProto) -> list[Mutation]:
        arrays = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        zero_inits = {name for name, arr in arrays.items() if arr.size and np.all(arr == 0)}
        muts = []
        for idx, node in enumerate(model.graph.node):
            if node.op_type != "Concat":
                continue
            zeros = [inp for inp in node.input if inp in zero_inits]
            if not zeros:
                continue
            first = zeros[0]
            for z in zeros[1:]:
                muts.append(
                    Mutation(
                        id=mutation_id(self.name, idx, z, first),
                        family=self.name,
                        description=f"reuse zero padding initializer {first} in Concat",
                        priority=170.0,
                        estimated_savings=int(arrays[z].size),
                        metadata={"kind": "reuse_zero", "node_index": idx, "old": z, "new": first},
                    )
                )
            if len(node.input) == 2 and len(zeros) == 1 and node.output:
                zero_name = zeros[0]
                zero_arr = arrays[zero_name]
                # ONNX Runtime rejects Pad on bool tensors in the opset used by
                # these task kits. Keep bool concat masks as concats.
                if zero_arr.dtype == np.bool_:
                    continue
                real_name = node.input[0] if node.input[1] == zero_name else node.input[1]
                axis = 0
                for attr in node.attribute:
                    if attr.name == "axis":
                        axis = int(attr.i)
                rank = zero_arr.ndim
                if rank > 0:
                    if axis < 0:
                        axis += rank
                    if 0 <= axis < rank:
                        pad_width = int(zero_arr.shape[axis])
                        before = node.input[0] == zero_name
                        muts.append(
                            Mutation(
                                id=mutation_id(self.name, "concat_to_pad", idx, node.output[0], axis, pad_width, before),
                                family=self.name,
                                description=f"replace Concat zero padding with Pad on axis {axis}",
                                priority=210.0 + pad_width,
                                estimated_savings=int(zero_arr.size),
                                metadata={
                                    "kind": "concat_zero_to_pad",
                                    "node_index": idx,
                                    "output": node.output[0],
                                    "real": real_name,
                                    "zero": zero_name,
                                    "axis": axis,
                                    "pad_width": pad_width,
                                    "before": before,
                                    "rank": rank,
                                },
                            )
                        )
        return muts[: task.config.max_candidates_per_family]

    def apply(self, task: TaskContext, model: onnx.ModelProto, mutation: Mutation) -> BuildResult:
        m = clone_model(model)
        if mutation.metadata.get("kind") == "concat_zero_to_pad":
            idx = int(mutation.metadata["node_index"])
            if idx >= len(m.graph.node) or m.graph.node[idx].op_type != "Concat":
                return BuildResult(model=None, ok=False, error="Concat node missing")
            output = mutation.metadata["output"]
            rank = int(mutation.metadata["rank"])
            pads = np.zeros(rank * 2, dtype=np.int64)
            offset = 0 if mutation.metadata["before"] else rank
            pads[offset + int(mutation.metadata["axis"])] = int(mutation.metadata["pad_width"])
            pads_name = f"{output}__optimizer_pads"
            value_name = f"{output}__optimizer_pad_value"
            zero_arr = None
            for init in m.graph.initializer:
                if init.name == mutation.metadata["zero"]:
                    zero_arr = numpy_helper.to_array(init)
                    break
            if zero_arr is None:
                return BuildResult(model=None, ok=False, error="zero initializer missing")
            m.graph.initializer.append(numpy_helper.from_array(pads, name=pads_name))
            m.graph.initializer.append(numpy_helper.from_array(np.array(0, dtype=zero_arr.dtype), name=value_name))
            m.graph.node[idx].output[0] = f"{output}__optimizer_dead"
            m.graph.node.append(helper.make_node("Pad", [mutation.metadata["real"], pads_name, value_name], [output], name=f"optimizer_pad_{idx}", mode="constant"))
            return self.build(task, m)
        old = mutation.metadata["old"]
        new = mutation.metadata["new"]
        for node in m.graph.node:
            for i, inp in enumerate(node.input):
                if inp == old:
                    node.input[i] = new
        return self.build(task, m)
