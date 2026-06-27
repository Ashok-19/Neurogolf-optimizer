from pathlib import Path
import unittest

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from neurogolf_optimizer.models import OptimizerConfig, TaskContext
from neurogolf_optimizer.mutations.initializer_dedup import InitializerDedup
from neurogolf_optimizer.mutations.broadcast_scalarize import BroadcastScalarize
from neurogolf_optimizer.mutations.gather_shrink import GatherShrink
from neurogolf_optimizer.mutations.dimension_sweep import DimensionSweep
from neurogolf_optimizer.mutations.concat_pad import ConcatPadRewrite
from neurogolf_optimizer.mutations.dtype_lowering import DtypeLowering
from neurogolf_optimizer.mutations.where_ablation import WhereAblation
from neurogolf_optimizer.mutations.where_combo import WhereCombo
from neurogolf_optimizer.mutations.where_simplify import WhereSimplify


def ctx() -> TaskContext:
    cfg = OptimizerConfig(repo_root=Path("."))
    return TaskContext("task000", Path("."), Path("base.onnx"), {}, Path("."), Path("."), Path("base.onnx"), 100, cfg)


def make_where_model():
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [1])
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    where = helper.make_node("Where", ["cond", "a", "b"], ["w"], name="w")
    out = helper.make_node("Identity", ["w"], ["y"], name="out")
    graph = helper.make_graph([where, out], "g", [cond, a, b], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])


class MutationGenerationTests(unittest.TestCase):
    def test_where_ablation_discovers_two_branches(self):
        muts = WhereAblation().discover(ctx(), make_where_model())
        self.assertEqual(len(muts), 2)
        result = WhereAblation().apply(ctx(), make_where_model(), muts[0])
        self.assertTrue(result.ok, result.error)

    def test_where_ablation_expands_broadcast_branch_for_consumers(self):
        cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [1, 2, 2])
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 2])
        y = helper.make_tensor_value_info("y", TensorProto.BOOL, [1, 2, 2])
        scalar = numpy_helper.from_array(np.array(0, dtype=np.float32), name="scalar_zero")
        w_info = helper.make_tensor_value_info("w", TensorProto.FLOAT, [1, 2, 2])
        where = helper.make_node("Where", ["cond", "scalar_zero", "x"], ["w"], name="w")
        eq = helper.make_node("Equal", ["w", "x"], ["y"], name="eq")
        graph = helper.make_graph([where, eq], "g", [cond, x], [y], [scalar], value_info=[w_info])
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])
        muts = [m for m in WhereAblation().discover(ctx(), model) if m.metadata["selected_input"] == "scalar_zero"]
        self.assertTrue(muts)
        self.assertTrue(muts[0].metadata["needs_expand"])
        result = WhereAblation().apply(ctx(), model, muts[0])
        self.assertTrue(result.ok, result.error)
        self.assertTrue(any(node.op_type == "Expand" for node in result.model.graph.node))

    def test_where_simplify_discovers_identical_branches(self):
        cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [1])
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
        where = helper.make_node("Where", ["cond", "x", "x"], ["y"], name="same")
        graph = helper.make_graph([where], "g", [cond, x], [y])
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])
        muts = WhereSimplify().discover(ctx(), model)
        self.assertEqual(len(muts), 1)
        result = WhereSimplify().apply(ctx(), model, muts[0])
        self.assertTrue(result.ok, result.error)
        self.assertTrue(any(node.op_type == "Identity" and node.output[0] == "y" for node in result.model.graph.node))

    def test_initializer_dedup_discovers_duplicates(self):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
        arr = np.array([1.0], dtype=np.float32)
        a = numpy_helper.from_array(arr, name="a")
        b = numpy_helper.from_array(arr, name="b")
        add = helper.make_node("Add", ["a", "b"], ["y"])
        graph = helper.make_graph([add], "g", [x], [y], [a, b])
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])
        muts = InitializerDedup().discover(ctx(), model)
        self.assertEqual(len(muts), 1)

    def test_where_combo_applies_multiple_choices(self):
        c = ctx()
        c.config.max_combo_size = 3
        model = make_where_model()
        # Add a second Where consuming the first identity output.
        model.graph.node.insert(1, helper.make_node("Where", ["cond", "a", "b"], ["w2"], name="w2"))
        model.graph.node[-1].input[0] = "w2"
        muts = WhereCombo().discover(c, model)
        self.assertTrue(any(len(m.metadata["choices"]) >= 2 for m in muts))
        result = WhereCombo().apply(c, model, next(m for m in muts if len(m.metadata["choices"]) >= 2))
        self.assertTrue(result.ok, result.error)

    def test_gather_shrink_remaps_indices(self):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
        table = numpy_helper.from_array(np.arange(12, dtype=np.float32).reshape(4, 3), name="table")
        idx = numpy_helper.from_array(np.array([0, 2], dtype=np.int64), name="idx")
        gather = helper.make_node("Gather", ["table", "idx"], ["y"], axis=0)
        graph = helper.make_graph([gather], "g", [x], [y], [table, idx])
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])
        muts = GatherShrink().discover(ctx(), model)
        self.assertTrue(muts)
        result = GatherShrink().apply(ctx(), model, muts[0])
        self.assertTrue(result.ok, result.error)
        inits = {i.name: numpy_helper.to_array(i) for i in result.model.graph.initializer}
        self.assertEqual(inits["table"].shape[0], 2)
        self.assertEqual(inits["idx"].tolist(), [0, 1])

    def test_dtype_lowering_keeps_reshape_shape_int64(self):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
        shape_init = numpy_helper.from_array(np.array([4], dtype=np.int64), name="shape")
        node = helper.make_node("Reshape", ["x", "shape"], ["y"])
        graph = helper.make_graph([node], "g", [x], [y], [shape_init])
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])
        muts = DtypeLowering().discover(ctx(), model)
        self.assertFalse(any(m.metadata.get("kind") == "int64_to_int32" for m in muts))

    def test_concat_pad_rewrite_uses_zero_initializer_dtype(self):
        x = helper.make_tensor_value_info("x", TensorProto.UINT8, [400])
        y = helper.make_tensor_value_info("y", TensorProto.UINT8, [445])
        zeros = numpy_helper.from_array(np.zeros(45, dtype=np.uint8), name="zeros45_u8")
        concat = helper.make_node("Concat", ["x", "zeros45_u8"], ["y"], axis=0)
        graph = helper.make_graph([concat], "g", [x], [y], [zeros])
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])
        muts = [m for m in ConcatPadRewrite().discover(ctx(), model) if m.metadata.get("kind") == "concat_zero_to_pad"]
        self.assertTrue(muts)
        result = ConcatPadRewrite().apply(ctx(), model, muts[0])
        self.assertTrue(result.ok, result.error)
        self.assertTrue(any(node.op_type == "Pad" for node in result.model.graph.node))
        inits = {init.name: numpy_helper.to_array(init) for init in result.model.graph.initializer}
        pad_values = [arr for name, arr in inits.items() if name.endswith("__optimizer_pad_value")]
        self.assertEqual(pad_values[0].dtype, np.uint8)

    def test_broadcast_scalarize_rewrites_uniform_where_branch(self):
        cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [1, 4, 4])
        value = helper.make_tensor_value_info("value", TensorProto.FLOAT, [1, 4, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 4])
        zeros = numpy_helper.from_array(np.zeros((1, 4, 4), dtype=np.float32), name="zeros")
        node = helper.make_node("Where", ["cond", "value", "zeros"], ["y"])
        graph = helper.make_graph([node], "g", [cond, value], [y], [zeros])
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])
        muts = BroadcastScalarize().discover(ctx(), model)
        self.assertEqual(len(muts), 1)
        result = BroadcastScalarize().apply(ctx(), model, muts[0])
        self.assertTrue(result.ok, result.error)
        inits = {init.name: numpy_helper.to_array(init) for init in result.model.graph.initializer}
        self.assertEqual(inits["zeros"].shape, ())
        self.assertEqual(float(inits["zeros"]), 0.0)

    def test_broadcast_scalarize_skips_concat_zero_padding(self):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 4])
        zeros = numpy_helper.from_array(np.zeros((1, 4, 4), dtype=np.float32), name="zeros")
        node = helper.make_node("Concat", ["x", "zeros"], ["y"], axis=1)
        graph = helper.make_graph([node], "g", [x], [y], [zeros])
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])
        self.assertFalse(BroadcastScalarize().discover(ctx(), model))

    def test_dimension_sweep_discovers_task014_axis_reduction(self):
        import onnx

        path = Path("task014/task/task014.onnx")
        if not path.exists():
            self.skipTest("task014 not present")
        c = ctx()
        c.task_id = "task014"
        c.task_dir = Path("task014")
        model = onnx.load(str(path))
        muts = DimensionSweep().discover(c, model)
        self.assertTrue(any(m.metadata.get("axis") == "row" and m.metadata.get("new_len") == 17 for m in muts))
        result = DimensionSweep().apply(c, model, next(m for m in muts if m.metadata.get("axis") == "row" and m.metadata.get("new_len") == 17))
        self.assertTrue(result.ok, result.error)


if __name__ == "__main__":
    unittest.main()
