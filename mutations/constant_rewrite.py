from __future__ import annotations

import onnx
from onnx import numpy_helper

from ..graph_utils import clone_model, initializer_signature
from ..models import BuildResult, Mutation, TaskContext
from ..mutation_base import BaseMutationFamily, mutation_id


class ConstantRewrite(BaseMutationFamily):
    name = "constant_rewrite"

    def discover(self, task: TaskContext, model: onnx.ModelProto) -> list[Mutation]:
        init_sigs = {initializer_signature(init): init.name for init in model.graph.initializer}
        const_sigs: dict[tuple, str] = {}
        muts = []
        for idx, node in enumerate(model.graph.node):
            if node.op_type != "Constant" or not node.output:
                continue
            tensor = None
            for attr in node.attribute:
                if attr.name == "value" and attr.HasField("t"):
                    tensor = attr.t
                    break
            if tensor is None:
                continue
            sig = initializer_signature(tensor)
            if sig in init_sigs:
                muts.append(
                    Mutation(
                        id=mutation_id(self.name, "const_to_existing", idx, node.output[0], init_sigs[sig]),
                        family=self.name,
                        description=f"reuse initializer {init_sigs[sig]} for Constant {node.output[0]}",
                        priority=240.0,
                        estimated_savings=int(numpy_helper.to_array(tensor).size),
                        metadata={"kind": "const_to_existing", "node_index": idx, "output": node.output[0], "initializer": init_sigs[sig]},
                    )
                )
            if sig in const_sigs:
                muts.append(
                    Mutation(
                        id=mutation_id(self.name, "merge_constant", idx, node.output[0], const_sigs[sig]),
                        family=self.name,
                        description=f"merge duplicate Constant {node.output[0]} into {const_sigs[sig]}",
                        priority=238.0,
                        estimated_savings=int(numpy_helper.to_array(tensor).size),
                        metadata={"kind": "merge_constant", "node_index": idx, "output": node.output[0], "replacement": const_sigs[sig]},
                    )
                )
            else:
                const_sigs[sig] = node.output[0]
            arr = numpy_helper.to_array(tensor)
            if arr.size == 1:
                for init_sig, init_name in init_sigs.items():
                    if init_sig[0] == sig[0] and init_sig[1] == sig[1] and init_sig[2] == sig[2]:
                        continue
        return muts

    def apply(self, task: TaskContext, model: onnx.ModelProto, mutation: Mutation) -> BuildResult:
        m = clone_model(model)
        old = mutation.metadata["output"]
        new = mutation.metadata.get("initializer") or mutation.metadata.get("replacement")
        for node in m.graph.node:
            for i, inp in enumerate(node.input):
                if inp == old:
                    node.input[i] = new
        return self.build(task, m)
