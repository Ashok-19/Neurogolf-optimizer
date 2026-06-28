from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

from neurogolf_optimizer.models import OptimizerConfig, ValidationResult
from neurogolf_optimizer.promotion import promote_candidate
from neurogolf_optimizer.repair_classification import FailureClass
from neurogolf_optimizer.repair_replay_campaign import (
    ReplayCandidate,
    prioritize_near_misses,
    run_replay,
    scan_near_misses,
)


def val(cost: int, public_wrong: int = 0, arc_wrong: int = 0, fresh_right: int = 200, fresh_total: int = 200) -> ValidationResult:
    return ValidationResult("task000", "m.onnx", 1, public_wrong, 1, arc_wrong, fresh_right, fresh_total, cost - 1, 1, cost, 1.0, public_wrong == 0 and arc_wrong == 0, 0, "{}", "", None)


def args(root: Path, **kwargs):
    data = {
        "task": [],
        "tasks": "task000",
        "max_tasks": None,
        "max_candidates": 10,
        "only_supported_failures": True,
        "fresh_min": 0.90,
        "require_fresh_exact": False,
        "require_exact": False,
        "allow_relaxed_fresh": True,
        "public_arc_only": False,
        "promote": False,
        "dry_run": True,
        "no_cache": True,
        "cache_dir": None,
        "clear_cache": False,
        "cache_stats": False,
        "timeout": 1200,
        "seed": 42,
        "allow_nonstrict_shapes": False,
        "keep_rejected": False,
        "delete_rejected": True,
        "delete_hopeless_rejected": True,
        "keep_diagnostic_rejected": True,
        "keep_near_miss_models": True,
        "keep_high_cost_probes": False,
        "validation_workers": 1,
        "validation_mode": "quick",
        "search_mode": "hybrid",
        "max_build_failures": 10000,
        "max_saved_models_per_task": 5000,
        "progress_every": 25,
        "time_limit_seconds": None,
        "run_id": "test_replay",
        "preset": None,
        "enable_synthesis": False,
        "synthesis_only": False,
        "synthesis_families": "",
        "max_programs": 2000,
        "max_python_candidates": 200,
        "max_onnx_compiles": 20,
        "fresh_gen_count": 0,
        "program_beam_size": 50,
        "dsl_max_depth": 4,
        "disable_waterfall_synthesis": False,
        "synthesis_max_depth": 3,
        "waterfall_beam_size": 80,
        "max_waterfall_programs": 1000,
        "allow_constant_output": False,
        "allow_public_specialization": False,
        "allow_bad_synth_probes": False,
        "allow_near_miss_synth": False,
        "disable_near_miss_repair": False,
        "max_near_miss_repairs": 8,
    }
    data.update(kwargs)
    ns = type("Args", (), data)()
    ns._root = root
    return ns


class PromotionReplayTests(unittest.TestCase):
    def test_promotion_helper_writes_notes_and_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task000"
            experiments = task_dir / "experiments"
            experiments.mkdir(parents=True)
            src = experiments / "new.onnx"
            target = experiments / "candidate.onnx"
            src.write_bytes(b"new")
            target.write_bytes(b"old")
            cfg = OptimizerConfig(repo_root=root, promote=True)
            with mock.patch("neurogolf_optimizer.promotion.validate_model", side_effect=[val(90), val(100)]):
                result = promote_candidate(task_dir, src, cfg, None, 100, baseline_cost=100, metadata={"repair_name": "r"})
            self.assertTrue(result.promoted)
            self.assertEqual(target.read_bytes(), b"new")
            self.assertTrue((experiments / "optimizer_accept_notes.md").exists())
            ledger = experiments / "optimizer_accept_ledger.jsonl"
            self.assertTrue(ledger.exists())
            row = json.loads(ledger.read_text().splitlines()[-1])
            self.assertEqual(row["old_cost"], 100)
            self.assertEqual(row["new_cost"], 90)
            self.assertEqual(row["repair_name"], "r")

    def test_promotion_helper_refuses_equal_current_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task000"
            experiments = task_dir / "experiments"
            experiments.mkdir(parents=True)
            src = experiments / "new.onnx"
            target = experiments / "candidate.onnx"
            src.write_bytes(b"new")
            target.write_bytes(b"old")
            cfg = OptimizerConfig(repo_root=root, promote=True)
            with mock.patch("neurogolf_optimizer.promotion.validate_model", side_effect=[val(100), val(100)]):
                result = promote_candidate(task_dir, src, cfg, None, 120, baseline_cost=120)
            self.assertFalse(result.promoted)
            self.assertEqual(result.reason, "cost_not_below_current_incumbent")
            self.assertEqual(target.read_bytes(), b"old")

    def test_promotion_helper_refuses_public_arc_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task000"
            experiments = task_dir / "experiments"
            experiments.mkdir(parents=True)
            src = experiments / "new.onnx"
            target = experiments / "candidate.onnx"
            src.write_bytes(b"new")
            target.write_bytes(b"old")
            cfg = OptimizerConfig(repo_root=root, promote=True)
            with mock.patch("neurogolf_optimizer.promotion.validate_model", return_value=val(90, public_wrong=1)):
                result = promote_candidate(task_dir, src, cfg, None, 100, baseline_cost=100)
            self.assertFalse(result.promoted)
            self.assertEqual(result.reason, "validation_not_accepted")
            self.assertEqual(target.read_bytes(), b"old")

    def test_near_miss_scanner_reads_json_and_jsonl_and_skips_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "task000/experiments/optimizer_runs/r1"
            run.mkdir(parents=True)
            model = run / "m.onnx"
            model.write_bytes(b"m")
            (run / "near_misses.json").write_text(json.dumps([{"task": "task000", "model_path": str(model), "candidate_cost": 90, "incumbent_cost": 100, "public_wrong": 0, "arc_gen_wrong": 1, "fresh_right": 200, "fresh_total": 200, "near_miss_category": "A"}]))
            (run / "near_misses.jsonl").write_text(json.dumps({"task": "task000", "model_path": str(run / "missing.onnx"), "candidate_cost": 80}) + "\n")
            rows = scan_near_misses(root, ["task000"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].candidate_cost, 90)

    def test_scanner_prioritizes_category_a_public_exact_low_arc(self):
        good = ReplayCandidate("task000", "a.onnx", 90, 100, 0, 1, 200, 200, "A", "r", [], [])
        bad = ReplayCandidate("task001", "b.onnx", 1, 100, 2, 99, 0, 200, "E", "r", [], [])
        self.assertIs(prioritize_near_misses([bad, good], 0.90)[0], good)

    def test_replay_runner_does_not_promote_in_dry_run_and_reports_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task000"
            task_dir.mkdir()
            model = task_dir / "m.onnx"
            model.write_bytes(b"m")
            candidate = ReplayCandidate("task000", str(model), 90, 100, 0, 1, 200, 200, "A", "r", [], [])
            repair = {"repair": "repair_ok", "accepted": True, "model_path": str(model), "validation": val(90).to_json_dict(), "repair_family": "internal_selector_extent"}
            with mock.patch("neurogolf_optimizer.repair_replay_campaign.scan_near_misses", return_value=[candidate]), \
                mock.patch("neurogolf_optimizer.repair_replay_campaign.config_from_args", return_value=OptimizerConfig(repo_root=root, no_cache=True, dry_run=True)), \
                mock.patch("neurogolf_optimizer.repair_replay_campaign.current_incumbent_cost", return_value=(None, 100, None)), \
                mock.patch("neurogolf_optimizer.repair_replay_campaign.read_baseline_cost", return_value=100), \
                mock.patch("neurogolf_optimizer.repair_replay_campaign.run_pairs", return_value={"failures": []}), \
                mock.patch("neurogolf_optimizer.repair_replay_campaign.classify_diagnostics", return_value=[FailureClass("missing_bottom_row", [2], [0], [1], [3, 1])]), \
                mock.patch("neurogolf_optimizer.repair_replay_campaign.try_repair_model", return_value=[repair]), \
                mock.patch("neurogolf_optimizer.repair_replay_campaign.promote_candidate") as promote:
                report = run_replay(args(root, tasks="task000", dry_run=True, run_id="dry"))
            self.assertEqual(report["summary"]["accepted_count"], 1)
            self.assertEqual(report["summary"]["promoted_count"], 0)
            self.assertFalse(promote.called)
            self.assertTrue((root / ".chatgpt/operations/neurogolf_optimizer/repair_replay_runs/dry/repair_replay_report.json").exists())

    def test_replay_runner_does_not_repromote_equal_current_incumbent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task000"
            task_dir.mkdir()
            model = task_dir / "m.onnx"
            model.write_bytes(b"m")
            candidate = ReplayCandidate("task000", str(model), 80, 100, 0, 1, 200, 200, "A", "r", [], [])
            repair = {"repair": "repair_equal", "accepted": True, "model_path": str(model), "validation": val(90).to_json_dict(), "repair_family": "internal_selector_extent"}
            with mock.patch("neurogolf_optimizer.repair_replay_campaign.scan_near_misses", return_value=[candidate]), \
                mock.patch("neurogolf_optimizer.repair_replay_campaign.config_from_args", return_value=OptimizerConfig(repo_root=root, no_cache=True, promote=True)), \
                mock.patch("neurogolf_optimizer.repair_replay_campaign.current_incumbent_cost", return_value=(None, 90, None)), \
                mock.patch("neurogolf_optimizer.repair_replay_campaign.read_baseline_cost", return_value=100), \
                mock.patch("neurogolf_optimizer.repair_replay_campaign.run_pairs", return_value={"failures": []}), \
                mock.patch("neurogolf_optimizer.repair_replay_campaign.classify_diagnostics", return_value=[FailureClass("missing_bottom_row", [2], [0], [1], [3, 1])]), \
                mock.patch("neurogolf_optimizer.repair_replay_campaign.try_repair_model", return_value=[repair]), \
                mock.patch("neurogolf_optimizer.repair_replay_campaign.promote_candidate") as promote:
                report = run_replay(args(root, tasks="task000", dry_run=False, promote=True, run_id="equal"))
            self.assertEqual(report["summary"]["accepted_count"], 0)
            self.assertEqual(report["summary"]["rejected_count"], 1)
            self.assertFalse(promote.called)


if __name__ == "__main__":
    unittest.main()
