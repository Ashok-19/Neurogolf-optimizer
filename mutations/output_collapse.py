from __future__ import annotations

import onnx

from ..graph_utils import clone_model, get_node_by_output
from ..models import BuildResult, Mutation, TaskContext
from ..mutation_base import BaseMutationFamily, mutation_id


class OutputCollapse(BaseMutationFamily):
    name = "output_collapse"

    def discover(self, task: TaskContext, model: onnx.ModelProto) -> list[Mutation]:
        if not model.graph.output:
            return []
        out = model.graph.output[0].name
        producer = get_node_by_output(model, out)
        if producer is None:
            return []
        muts = []
        if producer.op_type == "Where" and len(producer.input) >= 3:
            for idx, branch in ((1, "then"), (2, "else")):
                muts.append(
                    Mutation(
                        id=mutation_id(self.name, out, branch, producer.input[idx]),
                        family=self.name,
                        description=f"collapse final Where output to {branch} branch",
                        priority=900.0,
                        estimated_savings=None,
                        metadata={"kind": "final_where", "output": out, "selected": producer.input[idx], "branch": branch},
                    )
                )
        if producer.op_type == "Identity" and producer.input:
            muts.append(
                Mutation(
                    id=mutation_id(self.name, out, "identity", producer.input[0]),
                    family=self.name,
                    description="collapse final Identity",
                    priority=100.0,
                    estimated_savings=1,
                    metadata={"kind": "final_identity", "output": out, "selected": producer.input[0]},
                )
            )
        return muts

    def apply(self, task: TaskContext, model: onnx.ModelProto, mutation: Mutation) -> BuildResult:
        m = clone_model(model)
        output = mutation.metadata["output"]
        selected = mutation.metadata["selected"]
        for node in m.graph.node:
            for i, out in enumerate(node.output):
                if out == output:
                    node.output[i] = f"{output}__optimizer_dead"
        m.graph.node.append(onnx.helper.make_node("Identity", [selected], [output], name=f"optimizer_output_collapse_{mutation.metadata['kind']}"))
        return self.build(task, m)
