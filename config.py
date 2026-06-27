from __future__ import annotations

import argparse
from pathlib import Path

from .models import OptimizerConfig


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[3]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fresh-min", type=float, default=0.90)
    parser.add_argument("--require-fresh-exact", action="store_true")
    parser.add_argument("--require-exact", action="store_true")
    parser.add_argument("--allow-relaxed-fresh", action="store_true", default=True)
    parser.add_argument("--public-arc-only", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--cache-stats", action="store_true")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-nonstrict-shapes", action="store_true")
    parser.add_argument("--keep-rejected", action="store_true")
    parser.add_argument("--delete-rejected", action="store_true", default=True)
    parser.add_argument("--delete-hopeless-rejected", action="store_true", default=True)
    parser.add_argument("--keep-diagnostic-rejected", action="store_true", default=True)
    parser.add_argument("--keep-near-miss-models", action="store_true", default=True)
    parser.add_argument("--keep-high-cost-probes", action="store_true")
    parser.add_argument("--validation-workers", type=int, default=1)
    parser.add_argument("--validation-mode", choices=["quick", "official-first", "full"], default="quick")
    parser.add_argument("--search-mode", choices=["greedy", "beam", "evolutionary", "hybrid"], default="hybrid")
    parser.add_argument("--max-build-failures", type=int, default=10000)
    parser.add_argument("--max-saved-models-per-task", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--time-limit-seconds", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--preset", choices=["high-cost", "where-heavy", "table-heavy", "dtype-heavy", "conv-heavy", "known-good", "all"])
    parser.add_argument("--enable-synthesis", action="store_true")
    parser.add_argument("--synthesis-only", action="store_true")
    parser.add_argument("--synthesis-families", default="")
    parser.add_argument("--max-programs", type=int, default=2000)
    parser.add_argument("--max-python-candidates", type=int, default=200)
    parser.add_argument("--max-onnx-compiles", type=int, default=20)
    parser.add_argument("--fresh-gen-count", type=int, default=0)
    parser.add_argument("--program-beam-size", type=int, default=50)
    parser.add_argument("--dsl-max-depth", type=int, default=4)
    parser.add_argument("--disable-waterfall-synthesis", action="store_true")
    parser.add_argument("--synthesis-max-depth", type=int, default=3)
    parser.add_argument("--waterfall-beam-size", type=int, default=80)
    parser.add_argument("--max-waterfall-programs", type=int, default=1000)
    parser.add_argument("--allow-constant-output", action="store_true")
    parser.add_argument("--allow-public-specialization", action="store_true")
    parser.add_argument("--allow-bad-synth-probes", action="store_true")
    parser.add_argument("--allow-near-miss-synth", action="store_true")
    parser.add_argument("--disable-near-miss-repair", action="store_true")
    parser.add_argument("--max-near-miss-repairs", type=int, default=8)


def config_from_args(args: argparse.Namespace, repo_root: Path | None = None) -> OptimizerConfig:
    return OptimizerConfig(
        repo_root=repo_root or repo_root_from_file(),
        fresh_min=args.fresh_min,
        trials=getattr(args, "trials", getattr(args, "trials_per_task", 1000)),
        beam_size=getattr(args, "beam_size", 20),
        max_combo_size=getattr(args, "max_combo_size", 5),
        workers=getattr(args, "workers", 1),
        validation_workers=max(1, getattr(args, "validation_workers", 1)),
        timeout=args.timeout,
        promote=args.promote,
        dry_run=args.dry_run,
        require_exact=args.require_exact,
        require_fresh_exact=args.require_fresh_exact,
        allow_relaxed_fresh=args.allow_relaxed_fresh,
        public_arc_only=args.public_arc_only,
        no_cache=args.no_cache,
        cache_dir=args.cache_dir,
        clear_cache=args.clear_cache,
        cache_stats=args.cache_stats,
        seed=args.seed,
        max_saved_models_per_task=getattr(args, "max_saved_models_per_task", 5000),
        stop_after_no_improve=getattr(args, "stop_after_no_improve", 1000),
        target_gain=getattr(args, "target_gain", None),
        time_limit_seconds=getattr(args, "time_limit_seconds", None),
        allow_nonstrict_shapes=args.allow_nonstrict_shapes,
        validation_mode=args.validation_mode,
        search_mode=args.search_mode,
        families=[f for f in getattr(args, "families", "").split(",") if f],
        resume=getattr(args, "resume", False),
        run_id=getattr(args, "run_id", None),
        preset=getattr(args, "preset", None),
        keep_rejected=args.keep_rejected,
        delete_rejected=getattr(args, "delete_rejected", True),
        delete_hopeless_rejected=getattr(args, "delete_hopeless_rejected", True),
        keep_diagnostic_rejected=getattr(args, "keep_diagnostic_rejected", True),
        keep_near_miss_models=getattr(args, "keep_near_miss_models", True),
        keep_high_cost_probes=getattr(args, "keep_high_cost_probes", False),
        max_build_failures=getattr(args, "max_build_failures", 10000),
        progress_every=getattr(args, "progress_every", 25),
        enable_synthesis=getattr(args, "enable_synthesis", False) or getattr(args, "synthesis_only", False),
        synthesis_only=getattr(args, "synthesis_only", False),
        synthesis_families=[f for f in getattr(args, "synthesis_families", "").split(",") if f],
        max_programs=getattr(args, "max_programs", 2000),
        max_python_candidates=getattr(args, "max_python_candidates", 200),
        max_onnx_compiles=getattr(args, "max_onnx_compiles", 20),
        fresh_gen_count=getattr(args, "fresh_gen_count", 0),
        program_beam_size=getattr(args, "program_beam_size", 50),
        dsl_max_depth=getattr(args, "dsl_max_depth", 4),
        enable_waterfall_synthesis=not getattr(args, "disable_waterfall_synthesis", False),
        synthesis_max_depth=getattr(args, "synthesis_max_depth", 3),
        waterfall_beam_size=getattr(args, "waterfall_beam_size", 80),
        max_waterfall_programs=getattr(args, "max_waterfall_programs", 1000),
        allow_constant_output=getattr(args, "allow_constant_output", False),
        allow_public_specialization=getattr(args, "allow_public_specialization", False),
        allow_bad_synth_probes=getattr(args, "allow_bad_synth_probes", False),
        allow_near_miss_synth=getattr(args, "allow_near_miss_synth", False),
        enable_near_miss_repair=not getattr(args, "disable_near_miss_repair", False),
        max_near_miss_repairs=max(0, getattr(args, "max_near_miss_repairs", 8)),
    )
