from __future__ import annotations

import onnx

from ..graph_utils import clone_model, initializer_signature
from ..models import BuildResult, Mutation, TaskContext
from ..mutation_base import BaseMutationFamily, mutation_id


class InitializerDedup(BaseMutationFamily):
    name = "initializer_dedup"

    def discover(self, task: TaskContext, model: onnx.ModelProto) -> list[Mutation]:
        groups: dict[tuple[int, tuple[int, ...], str], list[str]] = {}
        for init in model.graph.initializer:
            groups.setdefault(initializer_signature(init), []).append(init.name)
        muts = []
        for sig, names in groups.items():
            if len(names) <= 1:
                continue
            keep = names[0]
            dups = names[1:]
            muts.append(
                Mutation(
                    id=mutation_id(self.name, keep, ",".join(dups)),
                    family=self.name,
                    description=f"deduplicate {len(dups)} initializers into {keep}",
                    priority=400.0 + len(dups),
                    estimated_savings=len(dups),
                    metadata={"keep": keep, "duplicates": dups, "signature": str(sig[:2])},
                )
            )
        return muts[: task.config.max_candidates_per_family]

    def apply(self, task: TaskContext, model: onnx.ModelProto, mutation: Mutation) -> BuildResult:
        m = clone_model(model)
        keep = mutation.metadata["keep"]
        duplicates = set(mutation.metadata["duplicates"])
        for node in m.graph.node:
            for i, inp in enumerate(node.input):
                if inp in duplicates:
                    node.input[i] = keep
        remaining = [init for init in m.graph.initializer if init.name not in duplicates]
        del m.graph.initializer[:]
        m.graph.initializer.extend(remaining)
        return self.build(task, m)
