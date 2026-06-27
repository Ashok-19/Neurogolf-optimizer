import unittest
from pathlib import Path

from neurogolf_optimizer.models import OptimizerConfig, TaskContext
from neurogolf_optimizer.mutations.synth_solver import SynthSolver
from neurogolf_optimizer.synthesis.candidate_program import ProgramCandidate
from neurogolf_optimizer.synthesis.dsl import Program, Step
from neurogolf_optimizer.synthesis.families import FAMILY_MODULES, generate_family_programs
from neurogolf_optimizer.synthesis.features import task_features
from neurogolf_optimizer.synthesis.onnx_emit import emit_program
from neurogolf_optimizer.synthesis.program_search import search_programs
from neurogolf_optimizer.synthesis.scoring import rank_program
from neurogolf_optimizer.synthesis.task_data import GridPair
from neurogolf_optimizer.synthesis.tests.test_grid import GridTests
from neurogolf_optimizer.synthesis.tests.test_objects import ObjectTests
from neurogolf_optimizer.synthesis.tests.test_dsl import DslTests
from neurogolf_optimizer.synthesis.tests.test_program_search import ProgramSearchTests
from neurogolf_optimizer.synthesis.tests.test_onnx_emit import OnnxEmitTests


class StrictSynthesisTests(unittest.TestCase):
    def test_synth_ranking_never_prioritizes_py_score_zero(self):
        exact = ProgramCandidate(Program("crop", "crop", steps=[Step("Input"), Step("CropNonZeroBBox")]), 2, 2, 2, 2, 0, 0, 2, estimated_cost=1000)
        wrong = ProgramCandidate(Program("id", "identity", steps=[Step("Input")]), 0, 2, 0, 2, 0, 0, 1, estimated_cost=1)
        self.assertGreater(rank_program(exact), rank_program(wrong))

    def test_public_arc_exact_candidate_ranks_before_cheap_wrong_candidate(self):
        exact = ProgramCandidate(Program("exact", "recolor", steps=[Step("Input")]), 3, 3, 5, 5, 0, 0, 5, estimated_cost=2000)
        cheap_wrong = ProgramCandidate(Program("cheap", "identity", steps=[Step("Input")]), 1, 3, 0, 5, 0, 0, 1, estimated_cost=1)
        self.assertGreater(rank_program(exact), rank_program(cheap_wrong))

    def test_empty_families_are_not_empty(self):
        pairs = [GridPair([[1, 0], [0, 0]], [[1, 0], [0, 0]], "train", 0)]
        feats = task_features(pairs)
        cfg = OptimizerConfig(repo_root=Path("."))
        checked = {
            "line_draw", "template_match", "gravity", "connected_components",
            "pattern_fill", "ray_projection", "sprite", "rectangle",
            "arithmetic_grid", "copy_paste", "scaling",
        }
        for family in checked:
            self.assertTrue(list(generate_family_programs(family, pairs, feats, cfg)), family)

    def test_task276_synth_recolor_not_emitted_if_worse(self):
        root = Path.cwd()
        task_dir = root / "task276"
        if not task_dir.exists():
            self.skipTest("task276 not present")
        cfg = OptimizerConfig(repo_root=root, enable_synthesis=True, synthesis_only=True, max_programs=200, max_python_candidates=50, max_onnx_compiles=20)
        ctx = TaskContext("task276", task_dir, task_dir / "task/task276.onnx", {}, task_dir / "experiments", task_dir / "experiments/optimizer_runs/test", task_dir / "task/task276.onnx", 10, cfg, baseline_cost=10)
        ctx.run_dir.mkdir(parents=True, exist_ok=True)
        muts = SynthSolver().discover(ctx, None)
        self.assertFalse(any((m.metadata.get("estimated_cost") or 0) >= 40522 for m in muts))

    def test_task031_crop_candidate_compilable_if_present(self):
        root = Path.cwd()
        task_dir = root / "task031"
        if not task_dir.exists():
            self.skipTest("task031 not present")
        cfg = OptimizerConfig(repo_root=root, enable_synthesis=True, synthesis_only=True, synthesis_families=["crop"], max_programs=500, max_python_candidates=50, max_onnx_compiles=20, keep_high_cost_probes=True)
        ctx = TaskContext("task031", task_dir, task_dir / "task/task031.onnx", {}, task_dir / "experiments", task_dir / "experiments/optimizer_runs/test", task_dir / "task/task031.onnx", 999999, cfg, baseline_cost=999999)
        ctx.run_dir.mkdir(parents=True, exist_ok=True)
        muts = SynthSolver().discover(ctx, None)
        self.assertTrue(any(m.family == "synth_solver" for m in muts))

    def test_task031_synthesis_does_not_try_identity_first(self):
        root = Path.cwd()
        task_dir = root / "task031"
        if not task_dir.exists():
            self.skipTest("task031 not present")
        cfg = OptimizerConfig(repo_root=root, enable_synthesis=True, synthesis_only=True, max_programs=1000, max_python_candidates=50)
        candidates = search_programs(task_dir, cfg)
        if not candidates:
            self.skipTest("task031 produced no strict synthesis candidates")
        first = candidates[0]
        self.assertGreater(first.public_right, 0)
        self.assertNotEqual(first.program.family, "identity")

    def test_waterfall_search_discovers_crop_then_recolor(self):
        import tempfile
        import onnx

        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp) / "task000"
            (task / "task").mkdir(parents=True)
            (task / "task" / "task000.json").write_text(
                '{"train":[{"input":[[0,0,0],[0,1,0],[0,0,0]],"output":[[2]]}],"test":[],"arc-gen":[]}'
            )
            cfg = OptimizerConfig(repo_root=Path(tmp), max_programs=300, max_python_candidates=50, max_waterfall_programs=300, synthesis_max_depth=2)
            candidates = search_programs(task, cfg)
            exact = [c for c in candidates if c.public_right == c.public_total and c.program.family == "waterfall"]
            self.assertTrue(any([s.op for s in c.program.steps] == ["Input", "CropColorBBox", "ReplaceColor"] for c in exact))
            onnx.checker.check_model(emit_program(exact[0].program))


if __name__ == "__main__":
    unittest.main()
