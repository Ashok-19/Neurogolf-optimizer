from __future__ import annotations

import numpy as np
import onnx
from onnx import numpy_helper

from ..graph_utils import clone_model, get_value_info_shapes, initializer_signature
from ..models import BuildResult, Mutation, TaskContext
from ..mutation_base import BaseMutationFamily, mutation_id


class WhereSimplify(BaseMutationFamily):
    name = "where_simplify"

    def discover(self, task: TaskContext, model: onnx.ModelProto) -> list[Mutation]:
        shapes = get_value_info_shapes(model)
        init_by_name = {init.name: init for init in model.graph.initializer}
        init_sigs = {}
        init_arrays = {}
        for name, init in init_by_name.items():
            try:
                init_sigs[name] = initializer_signature(init)
                init_arrays[name] = numpy_helper.to_array(init)
            except Exception:
                continue
        muts: list[Mutation] = []
        for idx, node in enumerate(model.graph.node):
            if node.op_type != "Where" or len(node.input) < 3 or not node.output:
                continue
            cond, then_value, else_value = node.input[:3]
            if then_value == else_value:
                if not self._replacement_shape_matches_output(shapes, then_value, node.output[0]):
                    continue
                muts.append(
                    self._mutation(
                        idx,
                        node,
                        then_value,
                        "identical_branches",
                        "replace Where with identical branch",
                        estimated_savings=1,
                        priority=380.0,
                    )
                )
                continue
            if then_value in init_sigs and else_value in init_sigs and init_sigs[then_value] == init_sigs[else_value]:
                if not self._replacement_shape_matches_output(shapes, then_value, node.output[0]):
                    continue
                arr = init_arrays.get(else_value)
                savings = 1 + (int(arr.size) if arr is not None else 0)
                muts.append(
                    self._mutation(
                        idx,
                        node,
                        then_value,
                        "duplicate_initializer_branches",
                        "replace Where with duplicate initializer branch",
                        estimated_savings=savings,
                        priority=378.0,
                    )
                )
                continue
            cond_arr = init_arrays.get(cond)
            if cond_arr is not None and cond_arr.size:
                if np.all(cond_arr):
                    if not self._replacement_shape_matches_output(shapes, then_value, node.output[0]):
                        continue
                    muts.append(
                        self._mutation(
                            idx,
                            node,
                            then_value,
                            "constant_true_condition",
                            "replace Where with then branch for all-true condition",
                            estimated_savings=int(cond_arr.size),
                            priority=376.0,
                        )
                    )
                elif not np.any(cond_arr):
                    if not self._replacement_shape_matches_output(shapes, else_value, node.output[0]):
                        continue
                    muts.append(
                        self._mutation(
                            idx,
                            node,
                            else_value,
                            "constant_false_condition",
                            "replace Where with else branch for all-false condition",
                            estimated_savings=int(cond_arr.size),
                            priority=376.0,
                        )
                    )
        return muts[: task.config.max_candidates_per_family]

    def _replacement_shape_matches_output(self, shapes: dict[str, list[int | None]], replacement: str, output: str) -> bool:
        replacement_shape = shapes.get(replacement)
        output_shape = shapes.get(output)
        if replacement_shape is None or output_shape is None:
            return False
        if any(dim is None for dim in replacement_shape + output_shape):
            return False
        return list(replacement_shape) == list(output_shape)

    def _mutation(
        self,
        idx: int,
        node: onnx.NodeProto,
        replacement: str,
        kind: str,
        description: str,
        estimated_savings: int,
        priority: float,
    ) -> Mutation:
        return Mutation(
            id=mutation_id(self.name, kind, idx, node.output[0], replacement),
            family=self.name,
            description=f"{description} at {node.output[0]}",
            priority=priority + estimated_savings / 1000.0,
            estimated_savings=estimated_savings,
            metadata={
                "kind": kind,
                "node_index": idx,
                "output": node.output[0],
                "replacement": replacement,
                "inputs": list(node.input),
            },
        )

    def apply(self, task: TaskContext, model: onnx.ModelProto, mutation: Mutation) -> BuildResult:
        m = clone_model(model)
        idx = int(mutation.metadata["node_index"])
        if idx >= len(m.graph.node) or m.graph.node[idx].op_type != "Where":
            return BuildResult(model=None, ok=False, error="Where node missing")
        node = m.graph.node[idx]
        output = mutation.metadata["output"]
        replacement = mutation.metadata["replacement"]
        if not node.output or node.output[0] != output:
            return BuildResult(model=None, ok=False, error="Where output changed")
        node.output[0] = f"{output}__optimizer_dead"
        identity = onnx.helper.make_node("Identity", [replacement], [output], name=f"optimizer_where_simplify_{idx}")
        m.graph.node.append(identity)
        return self.build(task, m)
