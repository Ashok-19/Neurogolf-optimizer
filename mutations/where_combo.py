from __future__ import annotations

from itertools import combinations

import onnx

from ..graph_utils import clone_model, get_consumers
from ..models import BuildResult, Mutation, TaskContext
from ..mutation_base import BaseMutationFamily, mutation_id
from .where_ablation import WhereAblation


class WhereCombo(BaseMutationFamily):
    name = "where_combo"

    def discover(self, task: TaskContext, model: onnx.ModelProto) -> list[Mutation]:
        singles = WhereAblation().discover(task, model)
        muts: list[Mutation] = []
        limit = min(len(singles), max(task.config.beam_size * 4, 12))
        pool = singles[:limit]
        max_size = min(task.config.max_combo_size, len(pool))
        for size in range(2, max_size + 1):
            for combo in combinations(pool, size):
                outputs = [m.metadata.get("output") for m in combo]
                if len(set(outputs)) != len(outputs):
                    continue
                choices = [m.metadata for m in combo]
                choice_metadata = [
                    {
                        "mutation_id": m.id,
                        "family": m.family,
                        "description": m.description,
                        "metadata": m.metadata,
                    }
                    for m in combo
                ]
                priority = sum(m.priority for m in combo) / size + size * 25
                savings = sum(m.estimated_savings or 0 for m in combo)
                muts.append(
                    Mutation(
                        id=mutation_id(self.name, tuple((c["output"], c["selected_branch"]) for c in choices)),
                        family=self.name,
                        description="combine " + "; ".join(m.description for m in combo),
                        priority=priority,
                        estimated_savings=savings,
                        metadata={
                            "choices": choices,
                            "choice_metadata": choice_metadata,
                            "single_ids": [m.id for m in combo],
                            "combo_size": size,
                        },
                    )
                )
                if len(muts) >= task.config.max_candidates_per_family:
                    break
            if len(muts) >= task.config.max_candidates_per_family:
                break
        muts.sort(key=lambda m: m.priority, reverse=True)
        return muts[: task.config.max_candidates_per_family]

    def apply(self, task: TaskContext, model: onnx.ModelProto, mutation: Mutation) -> BuildResult:
        m = clone_model(model)
        graph_outputs = {o.name for o in m.graph.output}
        for choice in mutation.metadata.get("choices", []):
            target = choice["output"]
            selected = choice.get("selected_input_name") or choice["selected_input"]
            node = None
            for n in m.graph.node:
                if n.op_type == "Where" and n.output and n.output[0] == target:
                    node = n
                    break
            if node is None:
                return BuildResult(model=None, ok=False, error=f"Where output {target} no longer exists")
            node.output[0] = f"{target}__optimizer_dead"
            selected_for_replacement = WhereAblation()._expanded_replacement(m, target, selected, choice, choice.get("node_index"))
            replacements = 0
            for consumer in get_consumers(m).get(target, []):
                for i, inp in enumerate(consumer.input):
                    if inp == target:
                        consumer.input[i] = selected_for_replacement
                        replacements += 1
            if target in graph_outputs:
                m.graph.node.append(
                    onnx.helper.make_node(
                        "Identity",
                        [selected_for_replacement],
                        [target],
                        name=f"optimizer_where_combo_{choice.get('node_index')}_{choice.get('selected_branch')}",
                    )
                )
                replacements += 1
            if replacements == 0:
                continue
        return self.build(task, m)
