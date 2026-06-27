from __future__ import annotations

import numpy as np
import onnx
from onnx import numpy_helper

from ..graph_utils import clone_model, get_consumers, get_producers, get_value_info_shapes
from ..models import BuildResult, Mutation, TaskContext
from ..mutation_base import BaseMutationFamily, mutation_id


class WhereAblation(BaseMutationFamily):
    name = "where_ablation"

    def discover(self, task: TaskContext, model: onnx.ModelProto) -> list[Mutation]:
        graph_outputs = {o.name for o in model.graph.output}
        producers = get_producers(model)
        shapes = self._shape_map(model)
        init_names = {i.name for i in model.graph.initializer}
        muts: list[Mutation] = []
        nodes = list(model.graph.node)
        for idx, node in enumerate(nodes):
            if node.op_type != "Where" or len(node.input) < 3 or not node.output:
                continue
            out = node.output[0]
            out_shape = shapes.get(out) or self._where_output_shape(node, shapes)
            for branch, selected_idx in (("then", 1), ("else", 2)):
                selected = node.input[selected_idx]
                dropped = node.input[3 - selected_idx]
                selected_shape = shapes.get(selected)
                if (
                    out_shape is not None
                    and selected_shape is not None
                    and selected_shape != out_shape
                    and not self._broadcastable(selected_shape, out_shape)
                ):
                    continue
                needs_expand = bool(out_shape is not None and selected_shape is not None and selected_shape != out_shape)
                dead_nodes, dead_inits = self._estimate_dead(model, dropped, selected, node.output[0], producers, init_names)
                output_bonus = 1000 if out in graph_outputs else 0
                priority = float(idx + output_bonus + dead_nodes * 5 + dead_inits * 10)
                muts.append(
                    Mutation(
                        id=mutation_id(self.name, idx, branch, out, selected),
                        family=self.name,
                        description=f"replace Where {out} with {branch} branch {selected}",
                        priority=priority,
                        estimated_savings=dead_nodes + dead_inits,
                        metadata={
                            "node_index": idx,
                            "node_name": node.name,
                            "output": out,
                            "selected_branch": branch,
                            "condition": node.input[0],
                            "condition_name": node.input[0],
                            "selected_input": selected,
                            "selected_input_name": selected,
                            "dropped_input": dropped,
                            "dropped_input_name": dropped,
                            "is_graph_output": out in graph_outputs,
                            "estimated_dead_nodes": dead_nodes,
                            "estimated_dead_initializers": dead_inits,
                            "selected_shape": selected_shape,
                            "output_shape": out_shape,
                            "needs_expand": needs_expand,
                        },
                    )
                )
        muts.sort(key=lambda m: m.priority, reverse=True)
        return muts[: task.config.max_candidates_per_family]

    def _shape_map(self, model: onnx.ModelProto) -> dict[str, list[int | None]]:
        shapes = get_value_info_shapes(model)
        try:
            inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
            inferred_shapes = get_value_info_shapes(inferred)
        except Exception:
            return shapes
        shapes.update(inferred_shapes)
        return shapes

    def _estimate_dead(self, model, dropped: str, selected: str, where_output: str, producers, init_names: set[str]) -> tuple[int, int]:
        consumers = get_consumers(model)
        if dropped == selected:
            return 0, 0
        protected = {selected, where_output}
        stack = [dropped]
        seen_values: set[str] = set()
        seen_nodes: set[int] = set()
        dead_inits: set[str] = set()
        node_index = {id(node): i for i, node in enumerate(model.graph.node)}
        while stack:
            value = stack.pop()
            if value in seen_values or value in protected:
                continue
            seen_values.add(value)
            if value in init_names:
                dead_inits.add(value)
            producer = producers.get(value)
            if producer is None:
                continue
            idx = node_index.get(id(producer))
            if idx is not None:
                seen_nodes.add(idx)
            for inp in producer.input:
                if inp:
                    stack.append(inp)
        return len(seen_nodes), len(dead_inits)

    def apply(self, task: TaskContext, model: onnx.ModelProto, mutation: Mutation) -> BuildResult:
        m = clone_model(model)
        target = mutation.metadata["output"]
        selected = mutation.metadata.get("selected_input_name") or mutation.metadata["selected_input"]
        node = None
        idx = mutation.metadata.get("node_index")
        if idx is not None and int(idx) < len(m.graph.node):
            candidate = m.graph.node[int(idx)]
            if candidate.op_type == "Where" and candidate.output and candidate.output[0] == target:
                node = candidate
        if node is None:
            for candidate in m.graph.node:
                if candidate.op_type == "Where" and candidate.output and candidate.output[0] == target:
                    node = candidate
                    break
        if node is None:
            return BuildResult(model=None, ok=False, error=f"Where output {target} no longer exists")
        if not node.output or node.output[0] != target:
            return BuildResult(model=None, ok=False, error="Where output changed")
        dead = f"{target}__optimizer_dead"
        node.output[0] = dead
        graph_outputs = {o.name for o in m.graph.output}
        selected_for_replacement = self._expanded_replacement(m, target, selected, mutation.metadata, idx)
        replacements = 0
        for consumer in get_consumers(m).get(target, []):
            for i, inp in enumerate(consumer.input):
                if inp == target:
                    consumer.input[i] = selected_for_replacement
                    replacements += 1
        if target in graph_outputs:
            identity = onnx.helper.make_node("Identity", [selected], [target], name=f"optimizer_where_{idx}_{mutation.metadata['selected_branch']}")
            identity.input[0] = selected_for_replacement
            m.graph.node.append(identity)
            replacements += 1
        if replacements == 0:
            return BuildResult(model=None, ok=False, error="Where output had no consumers")
        return self.build(task, m)

    def _expanded_replacement(self, model: onnx.ModelProto, target: str, selected: str, metadata: dict, idx) -> str:
        out_shape = metadata.get("output_shape")
        selected_shape = metadata.get("selected_shape")
        if out_shape is None or selected_shape is None or list(out_shape) == list(selected_shape):
            return selected
        if not self._broadcastable(list(selected_shape), list(out_shape)):
            return selected
        shape_name = f"{target}__optimizer_expand_shape_{idx}"
        expanded = f"{selected}__optimizer_expanded_for_{target}"
        existing_initializers = {init.name for init in model.graph.initializer}
        if shape_name not in existing_initializers:
            model.graph.initializer.append(numpy_helper.from_array(np.array(out_shape, dtype=np.int64), name=shape_name))
        existing_outputs = {out for node in model.graph.node for out in node.output}
        if expanded not in existing_outputs:
            model.graph.node.append(
                onnx.helper.make_node("Expand", [selected, shape_name], [expanded], name=f"optimizer_where_expand_{idx}")
            )
        return expanded

    def _broadcastable(self, src: list[int | None], dst: list[int | None]) -> bool:
        if any(x is None for x in src + dst):
            return False
        s = list(src)
        d = list(dst)
        if len(s) > len(d):
            return False
        s = [1] * (len(d) - len(s)) + s
        return all(a == b or a == 1 for a, b in zip(s, d))

    def _where_output_shape(self, node: onnx.NodeProto, shapes: dict[str, list[int | None]]) -> list[int | None] | None:
        inputs = []
        for value in node.input[:3]:
            shape = shapes.get(value)
            if shape is None or any(dim is None for dim in shape):
                return None
            inputs.append(tuple(int(dim) for dim in shape))
        try:
            return list(np.broadcast_shapes(*inputs))
        except ValueError:
            return None
