from pathlib import Path
import tempfile
import unittest

from neurogolf_optimizer.cache import ValidationCache
from neurogolf_optimizer.inspect_results import _rows
from neurogolf_optimizer.models import Candidate, OptimizerConfig, TaskContext, ValidationResult
from neurogolf_optimizer.optimizer import NeuroGolfOptimizer
from neurogolf_optimizer.reporting import write_json, write_run_report


class IntegrationControlTests(unittest.TestCase):
    def test_validation_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ValidationCache(Path(tmp))
            result = ValidationResult("task000", "m.onnx", 1, 0, 1, 0, 200, 200, 1, 2, 3, 24.0, True, 0, "{}", "", None)
            cache.put("task000", "abc", result)
            hit = cache.get("task000", "abc")
            self.assertIsNotNone(hit)
            self.assertEqual(hit.cost, 3)
            self.assertGreaterEqual(cache.stats()["entries"], 1)
            cache.close()

    def test_validation_cache_profile_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ValidationCache(Path(tmp))
            quick = ValidationResult("task000", "m.onnx", 1, 0, 1, 0, 10, 10, 1, 2, 3, 24.0, True, 0, "{}", "", None)
            full = ValidationResult("task000", "m.onnx", 1, 0, 1, 0, 200, 200, 4, 5, 9, 23.0, True, 0, "{}", "", None)
            cache.put("task000", "abc", quick, "quick")
            cache.put("task000", "abc", full, "full")
            self.assertEqual(cache.get("task000", "abc", "quick").cost, 3)
            self.assertEqual(cache.get("task000", "abc", "full").cost, 9)
            self.assertIsNone(cache.get("task000", "abc", "missing"))
            self.assertEqual(cache.stats()["profiles"], 2)
            cache.close()

    def test_report_does_not_overwrite_summary_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_json(run / "summary.json", {"detail": True})
            write_run_report(run, [], [], [])
            self.assertEqual((run / "summary.json").read_text().strip(), "{\n  \"detail\": true\n}")
            self.assertTrue((run / "campaign_summary.json").exists())

    def test_inspect_rows_reads_per_task_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "task066/experiments/optimizer_runs/r1"
            run.mkdir(parents=True)
            write_json(run / "summary.json", {"task": "task066", "accepted": [{"task": "task066", "accepted": True, "delta_points": 1.2, "candidate_cost": 10}], "near_misses": []})
            rows = _rows(root)
            self.assertTrue(any(r.get("task") == "task066" and r.get("accepted") for r in rows))

    def test_dry_run_promote_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiments = root / "task000/experiments"
            run = experiments / "optimizer_runs/r1"
            run.mkdir(parents=True)
            target = experiments / "candidate.onnx"
            target.write_bytes(b"old")
            src = run / "new.onnx"
            src.write_bytes(b"new")
            cfg = OptimizerConfig(repo_root=root, dry_run=True, promote=True)
            ctx = TaskContext("task000", root / "task000", root / "task000/task/task000.onnx", {}, experiments, run, src, 100, cfg, baseline_cost=100)
            opt = NeuroGolfOptimizer(cfg, run_id="r1")
            cand = Candidate("task000", "old", str(src), [], None, None, "h", 50, 1.0, True, False)
            self.assertFalse(opt._promote(ctx, cand, 100))
            self.assertEqual(target.read_bytes(), b"old")


if __name__ == "__main__":
    unittest.main()
