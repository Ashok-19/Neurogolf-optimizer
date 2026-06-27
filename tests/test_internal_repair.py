from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from neurogolf_optimizer.models import OptimizerConfig, ValidationResult
from neurogolf_optimizer.near_miss_lab import (
    _evaluate_repair,
    _internal_dimension_restore_model,
    _repair_cost_gate,
    try_repair_model,
)
from neurogolf_optimizer.repair_classification import FailureClass, classify_diagnostics


def missing_line_diagnostics(kind: str) -> dict:
    if kind == "col":
        return {
            "failures": [
                {
                    "source": "arc-gen",
                    "index": 14,
                    "expected_grid_shape": [7, 18],
                    "first_failing_coordinates": [
                        {
                            "row": row,
                            "col": 17,
                            "expected_tensor_active_channels": [5],
                            "predicted_tensor_active_channels": [],
                            "reason": ["missing_active_channels", "no_color"],
                        }
                        for row in range(7)
                    ],
                }
            ]
        }
    return {
        "failures": [
            {
                "source": "arc-gen",
                "index": 23,
                "expected_grid_shape": [7, 7],
                "first_failing_coordinates": [
                    {
                        "row": 6,
                        "col": col,
                        "expected_tensor_active_channels": [6],
                        "predicted_tensor_active_channels": [],
                        "reason": ["missing_active_channels", "no_color"],
                    }
                    for col in range(7)
                ],
            }
        ]
    }


def make_axis_restore_model():
    idx17 = numpy_helper.from_array(np.arange(17, dtype=np.int32), name="idx17_i32")
    min_c = numpy_helper.from_array(np.array([0], dtype=np.int32), name="min_c_1")
    min_r = numpy_helper.from_array(np.array([0], dtype=np.int32), name="min_r_1")
    col_out = numpy_helper.from_array(np.arange(17, dtype=np.int32).reshape(1, 1, 1, 17), name="col_out18_i32")
    row_out = numpy_helper.from_array(np.arange(17, dtype=np.int32).reshape(1, 1, 17, 1), name="row_out18_i32")
    pads = numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 13, 13], dtype=np.int64), name="pad_to_30_i64")
    nodes = [
        helper.make_node("Add", ["idx17_i32", "min_c_1"], ["col_idx_raw"], name="col_idx_raw"),
        helper.make_node("Add", ["idx17_i32", "min_r_1"], ["row_idx_raw"], name="row_idx_raw"),
    ]
    out = helper.make_tensor_value_info("col_idx_raw", TensorProto.INT32, [17])
    graph = helper.make_graph(nodes, "axis_restore", [], [out], [idx17, min_c, min_r, col_out, row_out, pads])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])


def make_cost_one_model():
    value = numpy_helper.from_array(np.array([1], dtype=np.float32), name="value")
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", ["value"], ["y"], name="out")
    graph = helper.make_graph([node], "cost_one", [], [y], [value])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])


class InternalRepairTests(unittest.TestCase):
    def test_failure_classification_detects_missing_last_column(self):
        classes = classify_diagnostics(missing_line_diagnostics("col"))
        self.assertEqual(classes[0].kind, "missing_last_column")
        self.assertEqual(classes[0].cols, [17])
        self.assertEqual(classes[0].expected_shape, [7, 18])

    def test_failure_classification_detects_missing_bottom_row(self):
        classes = classify_diagnostics(missing_line_diagnostics("row"))
        self.assertEqual(classes[0].kind, "missing_bottom_row")
        self.assertEqual(classes[0].rows, [6])
        self.assertEqual(classes[0].expected_shape, [7, 7])

    def test_internal_repair_path_is_ordered_before_dense_fallbacks(self):
        calls = []

        def producer(name):
            def inner(*_args, **_kwargs):
                calls.append(name)
                return [{"repair": name}]

            return inner

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("neurogolf_optimizer.near_miss_lab._try_internal_selector_extent_repairs", side_effect=producer("internal")), \
                mock.patch("neurogolf_optimizer.near_miss_lab._try_symbolic_line_repairs", side_effect=producer("symbolic")), \
                mock.patch("neurogolf_optimizer.near_miss_lab._try_mask_repairs", side_effect=producer("mask")), \
                mock.patch("neurogolf_optimizer.near_miss_lab._try_tiny_correction_repairs", side_effect=producer("tiny")), \
                mock.patch("neurogolf_optimizer.near_miss_lab._try_cleanup_repair", side_effect=producer("cleanup")):
                repairs = try_repair_model(
                    Path(tmp) / "task000",
                    Path("model.onnx"),
                    Path(tmp) / "repairs",
                    OptimizerConfig(repo_root=Path(tmp)),
                    mock.Mock(),
                    100,
                    diagnostics={"failures": []},
                )
        self.assertEqual(calls, ["internal", "symbolic", "mask", "tiny", "cleanup"])
        self.assertEqual([r["repair"] for r in repairs], calls)

    def test_primary_internal_repair_does_not_create_full_output_initializers(self):
        failure = FailureClass(kind="missing_last_column", rows=list(range(7)), cols=[17], channels=[5], expected_shape=[7, 18])
        repaired, meta = _internal_dimension_restore_model(make_axis_restore_model(), failure, "repair")
        self.assertIsNotNone(repaired, meta)
        shapes = [tuple(int(dim) for dim in init.dims) for init in repaired.graph.initializer]
        self.assertNotIn((1, 10, 30, 30), shapes)
        inits = {init.name: numpy_helper.to_array(init) for init in repaired.graph.initializer}
        self.assertEqual(inits["col_out18_i32"].shape, (1, 1, 1, 18))
        self.assertEqual(inits["row_out18_i32"].shape, (1, 1, 17, 1))
        self.assertTrue(any(name.startswith("idx17_i32_col_18") for name in inits))
        self.assertEqual(inits["pad_to_30_i64"].tolist(), [0, 0, 0, 0, 0, 0, 13, 12])

    def test_internal_repair_prefers_actual_dimension_sweep_provenance(self):
        failure = FailureClass(kind="missing_last_column", rows=list(range(7)), cols=[17], channels=[5], expected_shape=[7, 18])
        source = [
            {
                "id": "dimension_sweep:abc",
                "family": "dimension_sweep",
                "metadata": {"axis": "col", "old_len": 18, "new_len": 17, "source_initializer": "idx18_i32"},
            }
        ]
        _repaired, meta = _internal_dimension_restore_model(make_axis_restore_model(), failure, "repair", source_provenance=source)
        self.assertEqual(meta["source_mutation_family"], "dimension_sweep")
        self.assertEqual(meta["source_mutation"]["id"], "dimension_sweep:abc")

    def test_cost_gate_marks_obvious_dense_final_output_repairs_as_fallback(self):
        dense = numpy_helper.from_array(np.zeros((1, 10, 30, 30), dtype=np.float32), name="dense_patch")
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 10, 30, 30])
        node = helper.make_node("Identity", ["dense_patch"], ["y"], name="out")
        graph = helper.make_graph([node], "dense", [], [y], [dense])
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])
        gate = _repair_cost_gate(model, incumbent_cost=100, internal_or_dense="dense_final_output")
        self.assertFalse(gate["primary_repair"])
        self.assertIn("full_output_dense_initializer", gate["reason"])

    def test_cost_gate_marks_equal_cost_as_non_primary(self):
        gate = _repair_cost_gate(make_cost_one_model(), incumbent_cost=1, internal_or_dense="internal")
        self.assertFalse(gate["primary_repair"])
        self.assertIn("estimated_cost_not_below_incumbent", gate["reason"])

    def test_evaluate_repair_skips_non_primary_without_high_cost_probes(self):
        cfg = OptimizerConfig(repo_root=Path("."), keep_high_cost_probes=False)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("neurogolf_optimizer.near_miss_lab.run_pairs") as run_pairs, \
                mock.patch("neurogolf_optimizer.near_miss_lab.validate_model") as validate:
                row = _evaluate_repair(
                    Path(tmp) / "task000",
                    make_cost_one_model(),
                    "repair_equal_cost",
                    Path(tmp),
                    cfg,
                    mock.Mock(),
                    incumbent_cost=1,
                    repair_family="internal_selector_extent",
                    failure_class="missing_last_column",
                    internal_or_dense="internal",
                    repair_metadata={"source_mutations": [{"id": "m1", "family": "dimension_sweep", "metadata": {}}]},
                )
        self.assertTrue(row["ok"])
        self.assertTrue(row["skipped"])
        self.assertFalse(row["primary_repair"])
        self.assertIn("estimated_cost_not_below_incumbent", row["skip_reason"])
        self.assertIsNone(row["validation"])
        self.assertFalse(run_pairs.called)
        self.assertFalse(validate.called)
        self.assertEqual(row["source_mutation_families"], ["dimension_sweep"])
        self.assertEqual(row["source_mutation_ids"], ["m1"])

    def test_evaluate_repair_validates_non_primary_with_high_cost_probes(self):
        cfg = OptimizerConfig(repo_root=Path("."), keep_high_cost_probes=True)
        validation = ValidationResult("task000", "old.onnx", 1, 0, 1, 0, 200, 200, 0, 1, 1, 1.0, True, 0, "{}", "", None)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("neurogolf_optimizer.near_miss_lab.run_pairs", return_value={"failures": []}) as run_pairs, \
                mock.patch("neurogolf_optimizer.near_miss_lab.validate_model", return_value=validation) as validate:
                row = _evaluate_repair(
                    Path(tmp) / "task000",
                    make_cost_one_model(),
                    "repair_equal_cost",
                    Path(tmp),
                    cfg,
                    mock.Mock(),
                    incumbent_cost=1,
                    repair_family="internal_selector_extent",
                    failure_class="missing_last_column",
                    internal_or_dense="internal",
                    repair_metadata={},
                )
        self.assertTrue(row["ok"])
        self.assertFalse(row.get("skipped", False))
        self.assertFalse(row["primary_repair"])
        self.assertTrue(run_pairs.called)
        self.assertTrue(validate.called)
        self.assertEqual(row["actual_validation_cost"], 1)


if __name__ == "__main__":
    unittest.main()
