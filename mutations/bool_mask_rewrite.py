from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..graph_utils import clone_model, get_tensor_elem_type
from ..models import BuildResult, Mutation, TaskContext
from ..mutation_base import BaseMutationFamily, mutation_id


class BoolMaskRewrite(BaseMutationFamily):
    name = "bool_mask_rewrite"

    def discover(self, task: TaskContext, model: onnx.ModelProto) -> list[Mutation]:
        zero_inits = set()
        one_inits = set()
        for init in model.graph.initializer:
            try:
                arr = numpy_helper.to_array(init)
            except Exception:
                continue
            if arr.size and np.all(arr == 0):
                zero_inits.add(init.name)
            if arr.size and np.all(arr == 1):
                one_inits.add(init.name)
        muts = []
        for idx, node in enumerate(model.graph.node):
            if node.op_type == "Where" and len(node.input) >= 3 and len(node.output) == 1:
                for branch_idx in (1, 2):
                    other = 3 - branch_idx
                    if node.input[other] in zero_inits:
                        value_type = get_tensor_elem_type(model, node.input[branch_idx])
                        cond_type = get_tensor_elem_type(model, node.input[0])
                        if value_type == TensorProto.BOOL and cond_type == TensorProto.BOOL:
                            muts.append(
                                Mutation(
                                    id=mutation_id(self.name, "bool_and", idx, node.output[0], branch_idx),
                                    family=self.name,
                                    description=f"rewrite bool Where(mask, value, false) as And at {node.output[0]}",
                                    priority=230.0,
                                    estimated_savings=1,
                                    metadata={"kind": "bool_where_zero_to_and", "node_index": idx, "condition": node.input[0], "value": node.input[branch_idx], "zero": node.input[other], "output": node.output[0], "invert": branch_idx == 2},
                                )
                            )
                        muts.append(
                            Mutation(
                                id=mutation_id(self.name, idx, node.output[0], branch_idx),
                                family=self.name,
                                description=f"rewrite Where mask-zero pattern at {node.output[0]}",
                                priority=130.0,
                                estimated_savings=None,
                                metadata={"kind": "where_zero", "node_index": idx, "condition": node.input[0], "value": node.input[branch_idx], "zero": node.input[other], "output": node.output[0]},
                            )
                        )
                    if node.input[branch_idx] in one_inits and node.input[other] in zero_inits and get_tensor_elem_type(model, node.input[0]) == TensorProto.BOOL:
                        muts.append(
                            Mutation(
                                id=mutation_id(self.name, "where_true_false", idx, node.output[0], branch_idx),
                                family=self.name,
                                description=f"rewrite Where(mask, true, false) to mask at {node.output[0]}",
                                priority=235.0,
                                estimated_savings=1,
                                metadata={"kind": "where_true_false_to_mask", "node_index": idx, "condition": node.input[0], "output": node.output[0], "invert": branch_idx == 2},
                            )
                        )
        return muts[: task.config.max_candidates_per_family]

    def apply(self, task: TaskContext, model: onnx.ModelProto, mutation: Mutation) -> BuildResult:
        m = clone_model(model)
        kind = mutation.metadata["kind"]
        if kind == "where_true_false_to_mask":
            idx = mutation.metadata["node_index"]
            if idx >= len(m.graph.node) or m.graph.node[idx].op_type != "Where":
                return BuildResult(model=None, ok=False, error="node no longer Where")
            out = mutation.metadata["output"]
            src = mutation.metadata["condition"]
            m.graph.node[idx].output[0] = f"{out}__optimizer_dead"
            if mutation.metadata.get("invert"):
                src2 = f"{src}__optimizer_not"
                m.graph.node.append(helper.make_node("Not", [src], [src2], name=f"optimizer_mask_not_{idx}"))
                src = src2
            m.graph.node.append(helper.make_node("Identity", [src], [out], name=f"optimizer_mask_identity_{idx}"))
            return self.build(task, m)
        if kind == "bool_where_zero_to_and":
            idx = mutation.metadata["node_index"]
            if idx >= len(m.graph.node) or m.graph.node[idx].op_type != "Where":
                return BuildResult(model=None, ok=False, error="node no longer Where")
            cond = mutation.metadata["condition"]
            if mutation.metadata.get("invert"):
                inv = f"{cond}__optimizer_not"
                m.graph.node.append(helper.make_node("Not", [cond], [inv], name=f"optimizer_mask_not_{idx}"))
                cond = inv
            out = mutation.metadata["output"]
            m.graph.node[idx].output[0] = f"{out}__optimizer_dead"
            m.graph.node.append(helper.make_node("And", [cond, mutation.metadata["value"]], [out], name=f"optimizer_mask_and_{idx}"))
            return self.build(task, m)
        idx = mutation.metadata["node_index"]
        if idx >= len(m.graph.node):
            return BuildResult(model=None, ok=False, error="node missing")
        node = m.graph.node[idx]
        if node.op_type != "Where":
            return BuildResult(model=None, ok=False, error="node no longer Where")
        cast_out = f"{mutation.metadata['condition']}__optimizer_cast_like"
        cast = helper.make_node("Cast", [mutation.metadata["condition"]], [cast_out], name=f"optimizer_mask_cast_{idx}", to=TensorProto.FLOAT)
        mul = helper.make_node("Mul", [cast_out, mutation.metadata["value"]], [mutation.metadata["output"]], name=f"optimizer_mask_mul_{idx}")
        node.output[0] = f"{mutation.metadata['output']}__optimizer_dead"
        m.graph.node.extend([cast, mul])
        return self.build(task, m)
