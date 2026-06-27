from pathlib import Path
import unittest

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from neurogolf_optimizer.models import OptimizerConfig, TaskContext
from neurogolf_optimizer.pattern_miner import model_pattern_hits, rank_pattern_hits


def ctx() -> TaskContext:
    cfg = OptimizerConfig(repo_root=Path("."))
    return TaskContext("task000", Path("."), Path("base.onnx"), {}, Path("."), Path("."), Path("base.onnx"), 100, cfg)


class PatternMinerTests(unittest.TestCase):
    def test_model_pattern_hits_detect_concat_zero_tail(self):
        x = helper.make_tensor_value_info("x", TensorProto.UINT8, [4])
        y = helper.make_tensor_value_info("y", TensorProto.UINT8, [8])
        zeros = numpy_helper.from_array(np.zeros(4, dtype=np.uint8), name="zeros")
        node = helper.make_node("Concat", ["x", "zeros"], ["y"], axis=0)
        graph = helper.make_graph([node], "g", [x], [y], [zeros])
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])

        hits = model_pattern_hits("task000", Path("m.onnx"), model, ctx(), include_family_discovery=False)

        concat_hits = [hit for hit in hits if hit.pattern_id == "concat_zero_tail_uint8"]
        self.assertEqual(len(concat_hits), 1)
        self.assertEqual(concat_hits[0].rewrite_family, "concat_pad")

    def test_rank_pattern_hits_aggregates_tasks_and_sentinel(self):
        def model_for(task_id: str):
            cond = helper.make_tensor_value_info(f"cond_{task_id}", TensorProto.BOOL, [1, 4, 4])
            value = helper.make_tensor_value_info(f"value_{task_id}", TensorProto.FLOAT, [1, 4, 4])
            y = helper.make_tensor_value_info(f"y_{task_id}", TensorProto.FLOAT, [1, 4, 4])
            zeros = numpy_helper.from_array(np.zeros((1, 4, 4), dtype=np.float32), name=f"zeros_{task_id}")
            node = helper.make_node("Where", [cond.name, value.name, zeros.name], [y.name])
            graph = helper.make_graph([node], "g", [cond, value], [y], [zeros])
            return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])

        hits = []
        for task_id in ("task001", "task002"):
            c = ctx()
            c.task_id = task_id
            hits.extend(model_pattern_hits(task_id, Path(f"{task_id}.onnx"), model_for(task_id), c, include_family_discovery=True))

        opportunities = rank_pattern_hits(hits)
        scalar = next(item for item in opportunities if item.pattern_id == "mutation_family:broadcast_scalarize:broadcast_scalarize")
        self.assertEqual(scalar.affected_task_count, 2)
        self.assertIn(scalar.sentinel_task, {"task001", "task002"})
        self.assertEqual(scalar.rewrite_family, "broadcast_scalarize")


if __name__ == "__main__":
    unittest.main()
