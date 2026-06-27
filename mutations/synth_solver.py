from __future__ import annotations

import onnx

from ..models import BuildResult, Mutation, TaskContext
from ..mutation_base import BaseMutationFamily, mutation_id
from ..synthesis.dsl import Program
from ..synthesis.onnx_emit import emit_program
from ..synthesis.onnx_templates import estimate_model_cost, static_shape_preflight
from ..synthesis.program_search import search_programs
from ..reporting import write_json


class SynthSolver(BaseMutationFamily):
    name = "synth_solver"

    def discover(self, task: TaskContext, model: onnx.ModelProto) -> list[Mutation]:
        if not task.config.enable_synthesis:
            return []
        families = task.config.synthesis_families or None
        candidates = search_programs(task.task_dir, task.config, families=families)
        muts: list[Mutation] = []
        compiled = 0
        compiler_supported = 0
        skipped_worse = 0
        compile_failures: dict[str, int] = {}
        static_shape_failures: dict[str, int] = {}
        zero_score_skipped = 0
        python_exact = 0
        for cand in candidates:
            if cand.public_right == cand.public_total and cand.arc_right == cand.arc_total:
                python_exact += 1
            if cand.correctness <= 0.0:
                zero_score_skipped += 1
                continue
            if compiled >= task.config.max_onnx_compiles:
                break
            try:
                # Compile probe here so discovery only queues backend-supported programs.
                emitted = emit_program(cand.program)
            except Exception as exc:
                compile_failures[type(exc).__name__] = compile_failures.get(type(exc).__name__, 0) + 1
                continue
            compiler_supported += 1
            static_ok, static_error = static_shape_preflight(emitted)
            if not static_ok:
                key = static_error or "unknown static-shape failure"
                static_shape_failures[key] = static_shape_failures.get(key, 0) + 1
                continue
            estimated = estimate_model_cost(emitted)
            if estimated >= task.incumbent_cost and not task.config.keep_high_cost_probes:
                skipped_worse += 1
                continue
            compiled += 1
            priority = 1000.0 + cand.correctness * 100.0 - cand.complexity
            muts.append(
                Mutation(
                    id=mutation_id(self.name, cand.program.to_dict()),
                    family=self.name,
                    description=f"synthesize {cand.program.describe()} py_score={cand.correctness:.3f}",
                    priority=priority,
                    estimated_savings=max(0, task.incumbent_cost - estimated),
                    metadata={
                        "program": cand.program.to_dict(),
                        "program_score": cand.to_dict(),
                        "estimated_cost": estimated,
                        "incumbent_cost": task.incumbent_cost,
                        "compiler_backend": "synthesis.onnx_emit",
                    },
                )
            )
        write_json(
            task.run_dir / "synthesis_discovery_metrics.json",
            {
                "synth_programs_generated": len(candidates),
                "synth_programs_python_exact": python_exact,
                "synth_programs_compilable": compiler_supported,
                "synth_candidates_queued": compiled,
                "synth_programs_skipped_worse_than_incumbent": skipped_worse,
                "synth_programs_skipped_zero_score": zero_score_skipped,
                "compiler_failures": compile_failures,
                "static_shape_failures": static_shape_failures,
                "python_verification_mode": "official_tensor_public_arc_and_fresh_grid_relaxed",
            },
        )
        return muts

    def apply(self, task: TaskContext, model: onnx.ModelProto, mutation: Mutation) -> BuildResult:
        try:
            program = Program.from_dict(mutation.metadata["program"])
            emitted = emit_program(program)
            return self.build(task, emitted)
        except Exception as exc:
            return BuildResult(model=None, ok=False, error=f"synthesis emit failed: {type(exc).__name__}: {exc}")
