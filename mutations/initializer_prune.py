from __future__ import annotations

import onnx

from ..graph_utils import clone_model, remove_unused_initializers, remove_unused_value_info
from ..models import BuildResult, Mutation, TaskContext
from ..mutation_base import BaseMutationFamily, mutation_id


class InitializerPrune(BaseMutationFamily):
    name = "initializer_prune"

    def discover(self, task: TaskContext, model: onnx.ModelProto) -> list[Mutation]:
        used = {inp for node in model.graph.node for inp in node.input if inp}
        unused = [init.name for init in model.graph.initializer if init.name not in used]
        stale_vi = len(model.graph.value_info)
        if not unused and stale_vi == 0:
            return []
        return [
            Mutation(
                id=mutation_id(self.name, ",".join(unused), stale_vi),
                family=self.name,
                description=f"remove {len(unused)} unused initializers and stale value_info",
                priority=500.0 + len(unused),
                estimated_savings=len(unused),
                metadata={"unused_initializers": unused, "value_info_count": stale_vi},
            )
        ]

    def apply(self, task: TaskContext, model: onnx.ModelProto, mutation: Mutation) -> BuildResult:
        m = clone_model(model)
        before = len(m.graph.initializer)
        remove_unused_initializers(m)
        remove_unused_value_info(m)
        return self.build(task, m, changed=before != len(m.graph.initializer) or bool(mutation.metadata.get("value_info_count")))
