from __future__ import annotations

import onnx

from ..graph_utils import clone_model, get_consumers
from ..models import BuildResult, Mutation, TaskContext
from ..mutation_base import BaseMutationFamily, mutation_id


class IdentityCleanup(BaseMutationFamily):
    name = "identity_cleanup"

    def discover(self, task: TaskContext, model: onnx.ModelProto) -> list[Mutation]:
        graph_outputs = {o.name for o in model.graph.output}
        muts = []
        for idx, node in enumerate(model.graph.node):
            if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1 and node.output[0] not in graph_outputs:
                muts.append(
                    Mutation(
                        id=mutation_id(self.name, idx, node.input[0], node.output[0]),
                        family=self.name,
                        description=f"bypass Identity {node.output[0]}",
                        priority=100.0,
                        estimated_savings=1,
                        metadata={"node_index": idx, "input": node.input[0], "output": node.output[0]},
                    )
                )
        return muts

    def apply(self, task: TaskContext, model: onnx.ModelProto, mutation: Mutation) -> BuildResult:
        m = clone_model(model)
        old = mutation.metadata["output"]
        new = mutation.metadata["input"]
        for consumer in get_consumers(m).get(old, []):
            for i, inp in enumerate(consumer.input):
                if inp == old:
                    consumer.input[i] = new
        return self.build(task, m)
