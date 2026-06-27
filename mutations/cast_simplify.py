from __future__ import annotations

import onnx

from ..graph_utils import clone_model, get_consumers, get_tensor_elem_type
from ..models import BuildResult, Mutation, TaskContext
from ..mutation_base import BaseMutationFamily, mutation_id


def _attr_int(node: onnx.NodeProto, name: str) -> int | None:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return None


class CastSimplify(BaseMutationFamily):
    name = "cast_simplify"

    def discover(self, task: TaskContext, model: onnx.ModelProto) -> list[Mutation]:
        producers = {out: node for node in model.graph.node for out in node.output}
        muts = []
        for idx, node in enumerate(model.graph.node):
            if node.op_type != "Cast" or len(node.input) != 1 or len(node.output) != 1:
                continue
            parent = producers.get(node.input[0])
            to_type = _attr_int(node, "to")
            input_type = get_tensor_elem_type(model, node.input[0])
            if to_type is not None and input_type == to_type:
                muts.append(
                    Mutation(
                        id=mutation_id(self.name, "same_type", idx, node.output[0], node.input[0]),
                        family=self.name,
                        description=f"remove Cast to same type for {node.output[0]}",
                        priority=270.0,
                        estimated_savings=1,
                        metadata={"kind": "bypass_cast", "node_index": idx, "replace": node.output[0], "with": node.input[0]},
                    )
                )
            if parent and parent.op_type == "Cast" and _attr_int(parent, "to") == to_type:
                muts.append(
                    Mutation(
                        id=mutation_id(self.name, "double", idx, node.output[0]),
                        family=self.name,
                        description=f"remove redundant Cast chain producing {node.output[0]}",
                        priority=260.0,
                        estimated_savings=1,
                        metadata={"kind": "double_cast", "node_index": idx, "replace": node.output[0], "with": parent.output[0]},
                    )
                )
            if parent and parent.op_type == "Cast" and to_type is not None and parent.input:
                muts.append(
                    Mutation(
                        id=mutation_id(self.name, "collapse_chain", idx, node.output[0], parent.input[0], to_type),
                        family=self.name,
                        description=f"collapse Cast(Cast(x), T) for {node.output[0]}",
                        priority=250.0,
                        estimated_savings=1,
                        metadata={"kind": "collapse_chain", "node_index": idx, "cast_input": parent.input[0], "cast_output": node.output[0]},
                    )
                )
        return muts

    def apply(self, task: TaskContext, model: onnx.ModelProto, mutation: Mutation) -> BuildResult:
        m = clone_model(model)
        if mutation.metadata.get("kind") == "collapse_chain":
            idx = int(mutation.metadata["node_index"])
            if idx >= len(m.graph.node) or m.graph.node[idx].op_type != "Cast":
                return BuildResult(model=None, ok=False, error="Cast node missing")
            m.graph.node[idx].input[0] = mutation.metadata["cast_input"]
            return self.build(task, m)
        old = mutation.metadata["replace"]
        new = mutation.metadata["with"]
        for consumer in get_consumers(m).get(old, []):
            for i, inp in enumerate(consumer.input):
                if inp == old:
                    consumer.input[i] = new
        return self.build(task, m)
