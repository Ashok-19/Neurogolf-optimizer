from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from neurogolf_optimizer.graph_utils import model_hash, prune_unreachable, replace_graph_output_with_identity, safe_build


def make_model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    c = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="c")
    dead = numpy_helper.from_array(np.array([2.0], dtype=np.float32), name="dead_init")
    add = helper.make_node("Add", ["x", "c"], ["y"], name="live_add")
    dead_node = helper.make_node("Add", ["x", "dead_init"], ["dead"], name="dead_add")
    graph = helper.make_graph([dead_node, add], "g", [x], [y], [c, dead])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])


class GraphUtilTests(unittest.TestCase):
    def test_prune_unreachable(self):
        model = make_model()
        pruned = prune_unreachable(model)
        self.assertEqual([n.name for n in pruned.graph.node], ["live_add"])
        self.assertEqual([i.name for i in pruned.graph.initializer], ["c"])

    def test_safe_build_and_hash(self):
        model = make_model()
        result = safe_build(prune_unreachable(model))
        self.assertTrue(result.ok, result.error)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.onnx"
            onnx.save(result.model, path)
            self.assertEqual(len(model_hash(path)), 64)

    def test_replace_graph_output_with_identity_preserves_output_name(self):
        model = make_model()
        replace_graph_output_with_identity(model, "x", "y")
        self.assertEqual(model.graph.output[0].name, "y")
        self.assertTrue(any(n.op_type == "Identity" and n.output[0] == "y" for n in model.graph.node))


if __name__ == "__main__":
    unittest.main()
