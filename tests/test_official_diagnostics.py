import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from neurogolf_optimizer.models import Candidate, OptimizerConfig, ValidationResult
from neurogolf_optimizer.near_miss_lab import _symbolic_missing_line_templates, _try_mask_repairs, _try_symbolic_line_repairs
from neurogolf_optimizer.optimizer import NeuroGolfOptimizer
from neurogolf_optimizer.pair_runner import _compare_pair, classify_tensor_diff
from neurogolf_optimizer.score_gain_campaign import _finalize_nonterminal_tasks, _write_score_report
from neurogolf_optimizer.synthesis.dsl import Program, Step
from neurogolf_optimizer.synthesis.onnx_templates import emit_crop_nonzero_bbox_fixed, onehot_tensor_from_grid, static_shape_preflight
from neurogolf_optimizer.synthesis.task_data import GridPair
from neurogolf_optimizer.synthesis.verifier import verify_program


class OfficialDiagnosticsTests(unittest.TestCase):
    def test_official_public_arc_uses_exact_one_hot_tensor_equality(self):
        pair = GridPair([[0]], [[0]], "train", 0)
        pred = onehot_tensor_from_grid([[0]])
        pred[0, 0, 0, 1] = 1.0
        passed, diff, official_exact, fresh_exact = _compare_pair(pred, pair, "official_public_arc")
        self.assertFalse(passed)
        self.assertFalse(official_exact)
        self.assertTrue(fresh_exact)
        self.assertEqual(diff["extra_active_channel_0_outside_expected_output_shape"], 1)

    def test_fresh_grid_can_use_missing_zero_grid_comparison(self):
        pair = GridPair([[1]], [[1]], "fresh_gen", 0)
        pred = onehot_tensor_from_grid([[1, 0]])
        passed, _diff, official_exact, fresh_exact = _compare_pair(pred, pair, "fresh_grid")
        self.assertTrue(passed)
        self.assertFalse(official_exact)
        self.assertTrue(fresh_exact)

    def test_no_color_outside_shape_is_all_zero_channels(self):
        tensor = onehot_tensor_from_grid([[0]])
        self.assertEqual(float(tensor[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(tensor[0, :, 0, 1].sum()), 0.0)

    def test_active_black_outside_shape_is_detected(self):
        pred = onehot_tensor_from_grid([[0]])
        pred[0, 0, 0, 1] = 1.0
        diff = classify_tensor_diff(pred, [[0]])
        self.assertEqual(diff["kind"], "official_tensor_mismatch")
        self.assertEqual(diff["extra_active_channel_0_outside_expected_output_shape"], 1)

    def test_program_verifier_splits_official_and_relaxed_modes(self):
        program = Program("pad", "test", steps=[Step("Input"), Step("PadTo30", {"fill": 0})])
        pair = GridPair([[1]], [[1]], "train", 0)
        official_right, _ = verify_program(program, [pair], mode="official_tensor")
        relaxed_right, _ = verify_program(program, [pair], mode="grid_relaxed")
        self.assertEqual(official_right, 0)
        self.assertEqual(relaxed_right, 1)

    def test_dynamic_crop_emitter_passes_static_shape_preflight(self):
        ok, error = static_shape_preflight(emit_crop_nonzero_bbox_fixed(30, 30))
        self.assertTrue(ok, error)

    def test_diagnostic_rejected_candidate_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "candidate.onnx"
            model_path.write_bytes(b"model")
            cfg = OptimizerConfig(repo_root=root)
            opt = NeuroGolfOptimizer(cfg, run_id="test")
            validation = ValidationResult("task000", str(model_path), 4, 0, 9, 1, 200, 200, 10, 10, 20, 22.0, False, 1, "{}", "", None)
            cand = Candidate("task000", "parent", str(model_path), [], validation, None, "hash", 20, 1.0, False, True, "A")
            opt._apply_rejection_retention(cand, 100)
            self.assertTrue(model_path.exists())
            self.assertIn("public_exact", cand.retention_reason)

    def test_score_gain_report_includes_missing_task_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            cfg = OptimizerConfig(repo_root=run_dir)
            task_status = {"task001": {"status": "completed"}, "task002": {"status": "timeout", "error": "hit limit"}}
            _write_score_report(run_dir, cfg, ["task001", "task002"], [{"task": "task001"}], [], [], [{"task": "task002", "status": "timeout"}], 7157.53, task_status)
            report = json.loads((run_dir / "score_gain_report.json").read_text())
            self.assertEqual(len(report["summaries"]), 2)
            self.assertEqual(report["task_status"]["task002"]["status"], "timeout")

    def test_campaign_finalizes_running_and_pending_tasks(self):
        task_status = {
            "task001": {"status": "completed"},
            "task002": {"status": "running"},
            "task003": {"status": "pending"},
        }
        summaries = [{"task": "task001", "status": "completed"}]
        errors = []
        finalized = _finalize_nonterminal_tasks(
            ["task001", "task002", "task003"],
            task_status,
            summaries,
            errors,
            running_status="interrupted",
            pending_status="timeout",
            error="hit global limit",
        )
        self.assertEqual(finalized, ["task002", "task003"])
        self.assertEqual(task_status["task002"]["status"], "interrupted")
        self.assertEqual(task_status["task003"]["status"], "timeout")
        self.assertEqual({row["task"] for row in errors}, {"task002", "task003"})

    def test_near_miss_lab_generates_non_cleanup_repairs(self):
        diagnostics = {"failures": [{"expected_grid_shape": [2, 3]}]}
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "task000"
            (task_dir / "task").mkdir(parents=True)
            (task_dir / "task" / "task000.json").write_text('{"train":[],"test":[],"arc-gen":[]}')
            with mock.patch("neurogolf_optimizer.near_miss_lab.load_model", return_value=object()), \
                mock.patch("neurogolf_optimizer.near_miss_lab._output_shape_mask_model", return_value=object()), \
                mock.patch("neurogolf_optimizer.near_miss_lab._channel0_outside_mask_model", return_value=object()), \
                mock.patch("neurogolf_optimizer.near_miss_lab._evaluate_repair", side_effect=lambda *args: {"repair": args[2]}):
                repairs = _try_mask_repairs(task_dir, Path("m.onnx"), diagnostics, Path(tmp), OptimizerConfig(repo_root=Path(tmp)), mock.Mock(), 100)
        self.assertTrue(any(r["repair"].startswith("repair_no_color_outside") for r in repairs))
        self.assertFalse(all(r["repair"] == "cleanup_safe_build" for r in repairs))

    def test_symbolic_missing_line_classifier_detects_last_column(self):
        diagnostics = {
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
        templates = _symbolic_missing_line_templates(diagnostics)
        self.assertEqual(templates[0]["kind"], "missing_last_column")
        self.assertEqual(templates[0]["cols"], [17])
        self.assertEqual(templates[0]["selector"], {"kind": "adjacent_left_active", "col": 16})
        self.assertEqual(templates[0]["channel_groups"][0]["channel"], 5)

    def test_symbolic_line_repair_runs_before_dense_mask_repairs(self):
        diagnostics = {
            "failures": [
                {
                    "expected_grid_shape": [2, 3],
                    "first_failing_coordinates": [
                        {
                            "row": 0,
                            "col": 2,
                            "expected_tensor_active_channels": [4],
                            "predicted_tensor_active_channels": [],
                            "reason": ["missing_active_channels", "no_color"],
                        },
                        {
                            "row": 1,
                            "col": 2,
                            "expected_tensor_active_channels": [4],
                            "predicted_tensor_active_channels": [],
                            "reason": ["missing_active_channels", "no_color"],
                        },
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("neurogolf_optimizer.near_miss_lab.load_model", return_value=object()), \
                mock.patch("neurogolf_optimizer.near_miss_lab._symbolic_missing_line_patch_model", return_value=object()), \
                mock.patch("neurogolf_optimizer.near_miss_lab._evaluate_repair", side_effect=lambda *args: {"repair": args[2]}):
                repairs = _try_symbolic_line_repairs(Path(tmp) / "task000", Path("m.onnx"), diagnostics, Path(tmp), OptimizerConfig(repo_root=Path(tmp)), mock.Mock(), 100)
        self.assertTrue(repairs)
        self.assertTrue(repairs[0]["repair"].startswith("repair_symbolic_missing_last_column"))

    def test_optimizer_adapts_near_miss_repairs_to_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task000"
            run_dir = task_dir / "experiments" / "optimizer_runs" / "r1"
            run_dir.mkdir(parents=True)
            source = run_dir / "source.onnx"
            repaired = run_dir / "repaired.onnx"
            source.write_bytes(b"source")
            repaired.write_bytes(b"repaired")
            cfg = OptimizerConfig(repo_root=root, fresh_min=0.90)
            ctx = opt_ctx = mock.Mock()
            ctx.task_id = "task000"
            ctx.task_dir = task_dir
            ctx.run_dir = run_dir
            ctx.config = cfg
            cand = Candidate("task000", "parent", str(source), [], None, None, "h0", 90, 1.0, False, True, "A")
            repair = {
                "ok": True,
                "repair": "repair_tiny_coordinate_patch_0",
                "model_path": str(repaired),
                "validation": {
                    "task": "task000",
                    "model_path": str(repaired),
                    "public_wrong": 0,
                    "arc_gen_wrong": 0,
                    "fresh_right": 200,
                    "fresh_total": 200,
                    "memory": 40,
                    "params": 5,
                    "cost": 45,
                    "points": 20.0,
                    "exact": True,
                    "returncode": 0,
                },
            }
            with mock.patch("neurogolf_optimizer.optimizer.try_repair_model", return_value=[repair]):
                out = NeuroGolfOptimizer(cfg, run_id="r1")._near_miss_repair_candidates(opt_ctx, cand, 100)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].accepted)
        self.assertEqual(out[0].mutations[-1].family, "near_miss_repair")


if __name__ == "__main__":
    unittest.main()
