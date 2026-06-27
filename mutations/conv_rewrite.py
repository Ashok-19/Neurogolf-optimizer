from __future__ import annotations

import numpy as np
import onnx
from onnx import numpy_helper

from ..graph_utils import clone_model, get_consumers, set_initializer
from ..models import BuildResult, Mutation, TaskContext
from ..mutation_base import BaseMutationFamily, mutation_id


class ConvRewrite(BaseMutationFamily):
    name = "conv_rewrite"

    def discover(self, task: TaskContext, model: onnx.ModelProto) -> list[Mutation]:
        arrays = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        muts = []
        for idx, node in enumerate(model.graph.node):
            if node.op_type not in {"Conv", "ConvTranspose"} or len(node.input) < 3:
                continue
            bias_name = node.input[2]
            bias = arrays.get(bias_name)
            if bias is not None and bias.size and np.all(bias == 0):
                muts.append(
                    Mutation(
                        id=mutation_id(self.name, "zero_bias", idx, bias_name),
                        family=self.name,
                        description=f"remove zero bias {bias_name} from {node.op_type}",
                        priority=160.0,
                        estimated_savings=int(bias.size),
                        metadata={"kind": "remove_zero_bias", "node_index": idx, "bias": bias_name},
                    )
                )
            weight = arrays.get(node.input[1]) if len(node.input) > 1 else None
            if node.op_type == "Conv" and weight is not None and weight.ndim >= 2 and node.output:
                for consumer in get_consumers(model).get(node.output[0], []):
                    if consumer.op_type != "Gather" or len(consumer.input) < 2 or consumer.input[0] != node.output[0]:
                        continue
                    axis = 0
                    for attr in consumer.attribute:
                        if attr.name == "axis":
                            axis = int(attr.i)
                    if axis not in (1, -3) or consumer.input[1] not in arrays:
                        continue
                    indices = arrays[consumer.input[1]]
                    if not np.issubdtype(indices.dtype, np.integer):
                        continue
                    used = indices.astype(np.int64)
                    used[used < 0] += weight.shape[0]
                    if used.size == 0 or used.min() < 0 or used.max() >= weight.shape[0]:
                        continue
                    used_list = sorted({int(x) for x in used.reshape(-1)})
                    if len(used_list) >= weight.shape[0]:
                        continue
                    savings = int((weight.shape[0] - len(used_list)) * np.prod(weight.shape[1:]))
                    muts.append(
                        Mutation(
                            id=mutation_id(self.name, "conv_gather_prune", idx, consumer.output[0], tuple(used_list)),
                            family=self.name,
                            description=f"prune Conv output channels used by Gather to {len(used_list)}/{weight.shape[0]}",
                            priority=260.0 + savings / 1000.0,
                            estimated_savings=savings,
                            metadata={"kind": "conv_gather_channel_prune", "node_index": idx, "weight": node.input[1], "bias": node.input[2] if len(node.input) > 2 else None, "gather_indices": consumer.input[1], "used": used_list},
                        )
                    )
        return muts

    def apply(self, task: TaskContext, model: onnx.ModelProto, mutation: Mutation) -> BuildResult:
        m = clone_model(model)
        if mutation.metadata.get("kind") == "conv_gather_channel_prune":
            used = list(mutation.metadata["used"])
            remap = {old: new for new, old in enumerate(used)}
            arrays = {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}
            weight = arrays.get(mutation.metadata["weight"])
            if weight is None:
                return BuildResult(model=None, ok=False, error="weight missing")
            set_initializer(m, mutation.metadata["weight"], weight[used])
            bias_name = mutation.metadata.get("bias")
            if bias_name and bias_name in arrays:
                set_initializer(m, bias_name, arrays[bias_name][used])
            idx_name = mutation.metadata["gather_indices"]
            idx = arrays[idx_name]
            norm = idx.astype(np.int64)
            norm[norm < 0] += weight.shape[0]
            new_idx = np.vectorize(lambda x: remap[int(x)])(norm).astype(idx.dtype)
            set_initializer(m, idx_name, new_idx)
            return self.build(task, m)
        idx = mutation.metadata["node_index"]
        if idx >= len(m.graph.node):
            return BuildResult(model=None, ok=False, error="node missing")
        node = m.graph.node[idx]
        if node.op_type not in {"Conv", "ConvTranspose"} or len(node.input) < 3:
            return BuildResult(model=None, ok=False, error="not a biased Conv")
        del node.input[2:]
        return self.build(task, m)
