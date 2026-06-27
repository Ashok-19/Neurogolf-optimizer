from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "neurogolf_optimizer"

from .cache import ValidationCache
from .config import add_common_args, config_from_args
from .graph_utils import save_model
from .reporting import write_json
from .scoring import delta_points, is_accepted
from .search.scheduler import discover_tasks, read_baseline_cost
from .synthesis.onnx_emit import emit_program
from .synthesis.onnx_templates import estimate_model_cost
from .synthesis.program_search import search_programs
from .validate import validate_model


def _task_list(args, config) -> list[str]:
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    elif args.all:
        tasks = sorted(p.name for p in config.repo_root.glob("task[0-9][0-9][0-9]") if (p / "validate_task.py").exists())
    else:
        tasks = discover_tasks(config.repo_root, all_tasks=False, tasks=None, top_cost=args.top_cost, min_cost=args.min_cost, preset=config.preset)
    if args.min_cost is not None:
        tasks = [t for t in tasks if read_baseline_cost(config.repo_root / t) >= args.min_cost]
    if args.top_cost is not None:
        tasks = tasks[: args.top_cost]
    return tasks


def _candidate_quality(cand) -> dict:
    public_wrong = cand.public_total - cand.public_right
    arc_wrong = cand.arc_total - cand.arc_right
    return {
        "name": cand.program.name,
        "family": cand.program.family,
        "description": cand.program.describe(),
        "public_right": cand.public_right,
        "public_total": cand.public_total,
        "public_wrong": public_wrong,
        "arc_right": cand.arc_right,
        "arc_total": cand.arc_total,
        "arc_wrong": arc_wrong,
        "fresh_right": cand.fresh_right,
        "fresh_total": cand.fresh_total,
        "python_correctness": cand.correctness,
        "estimated_cost": cand.estimated_cost,
        "compiled": cand.compiled,
        "compile_error": cand.compile_error,
        "program": cand.program.to_dict(),
    }


def _validate_candidates(task: str, task_dir: Path, config, cache: ValidationCache, out_dir: Path, candidates: list, validate_top_k: int, cheap_only: bool, exact_only: bool, incumbent_cost: int) -> tuple[list[dict], Counter]:
    validations: list[dict] = []
    failures: Counter = Counter()
    model_dir = task_dir / "experiments" / "synthesis_audit"
    model_dir.mkdir(parents=True, exist_ok=True)
    validated = 0
    for idx, cand in enumerate(candidates):
        if validated >= validate_top_k:
            break
        if exact_only and not (cand.public_right == cand.public_total and cand.arc_right == cand.arc_total):
            continue
        if cheap_only and (cand.estimated_cost is None or cand.estimated_cost >= incumbent_cost):
            continue
        if cand.compile_error:
            continue
        try:
            model = emit_program(cand.program)
            path = model_dir / f"audit_{int(time.time())}_{idx:03d}.onnx"
            save_model(model, path)
            result = validate_model(task_dir, path, config, cache)
            vdict = result.to_json_dict()
            validations.append(
                {
                    "program": _candidate_quality(cand),
                    "model_path": str(path),
                    "validation": vdict,
                    "accepted": is_accepted(vdict, incumbent_cost, config),
                    "positive_score_gain": result.cost is not None and result.cost < incumbent_cost,
                    "delta_points": None if result.cost is None else delta_points(incumbent_cost, result.cost),
                }
            )
            validated += 1
        except Exception as exc:
            reason = f"{type(exc).__name__}:{str(exc).splitlines()[0][:120]}"
            failures[reason] += 1
    return validations, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit synthesis backend quality and optional accepted-score gain")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--tasks")
    parser.add_argument("--top-cost", type=int)
    parser.add_argument("--min-cost", type=int)
    parser.add_argument("--validate-top-k", type=int, default=0)
    parser.add_argument("--validate-cheap-only", action="store_true")
    parser.add_argument("--validate-exact-only", action="store_true")
    add_common_args(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)
    tasks = _task_list(args, config)
    out_dir = config.repo_root / ".chatgpt/operations/neurogolf_optimizer/synthesis_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = ValidationCache(config.cache_dir or (config.repo_root / ".chatgpt/operations/neurogolf_optimizer/cache"), disabled=config.no_cache)
    family_hits: Counter = Counter()
    compiler_support: Counter = Counter()
    compiler_failures: Counter = Counter()
    validation_failures: Counter = Counter()
    task_hits: dict[str, dict] = {}
    rows: list[dict] = []
    counters = Counter()
    try:
        for task in tasks:
            task_dir = config.repo_root / task
            incumbent_cost = read_baseline_cost(task_dir)
            candidates = search_programs(task_dir, config, families=config.synthesis_families or None)
            if candidates:
                counters["tasks_with_any_python_candidate"] += 1
            public_exact = 0
            public_arc_exact = 0
            compilable = 0
            cheap_compilable = 0
            task_failures: Counter = Counter()
            enriched = []
            for cand in candidates:
                family_hits[cand.program.family] += 1
                public_wrong = cand.public_total - cand.public_right
                arc_wrong = cand.arc_total - cand.arc_right
                if public_wrong == 0:
                    public_exact += 1
                if public_wrong == 0 and arc_wrong == 0:
                    public_arc_exact += 1
                try:
                    model = emit_program(cand.program)
                    estimated = estimate_model_cost(model)
                    cand.estimated_cost = estimated
                    cand.compiled = True
                    compiler_support[cand.program.family] += 1
                    compilable += 1
                    if estimated < incumbent_cost or config.keep_high_cost_probes:
                        cheap_compilable += 1
                except Exception as exc:
                    reason = f"{type(exc).__name__}:{str(exc).splitlines()[0][:120]}"
                    cand.compile_error = reason
                    compiler_failures[reason] += 1
                    task_failures[reason] += 1
                enriched.append(cand)
            if public_exact:
                counters["tasks_with_python_public_exact"] += 1
            if public_arc_exact:
                counters["tasks_with_python_public_arc_exact"] += 1
            if compilable:
                counters["tasks_with_compilable_candidate"] += 1
            if cheap_compilable:
                counters["tasks_with_cheap_compilable_candidate"] += 1
            validations: list[dict] = []
            if args.validate_top_k > 0:
                validations, vf = _validate_candidates(
                    task,
                    task_dir,
                    config,
                    cache,
                    out_dir,
                    enriched,
                    args.validate_top_k,
                    args.validate_cheap_only,
                    args.validate_exact_only,
                    incumbent_cost,
                )
                validation_failures.update(vf)
                if validations:
                    counters["tasks_with_validated_candidate"] += 1
                if any(v["accepted"] for v in validations):
                    counters["tasks_with_accepted_candidate"] += 1
                if any(v["positive_score_gain"] for v in validations):
                    counters["tasks_with_positive_score_gain"] += 1
            best_validation = min(
                (v for v in validations if v.get("validation", {}).get("cost") is not None),
                key=lambda v: v["validation"]["cost"],
                default=None,
            )
            row = {
                "task": task,
                "baseline_cost": incumbent_cost,
                "python_candidates": len(candidates),
                "python_public_exact": public_exact,
                "python_public_arc_exact": public_arc_exact,
                "compilable_candidates": compilable,
                "cheap_compilable_candidates": cheap_compilable,
                "validated_candidates": len(validations),
                "accepted_candidates": sum(1 for v in validations if v["accepted"]),
                "best_validated_cost": None if best_validation is None else best_validation["validation"].get("cost"),
                "best_validated_public_wrong": None if best_validation is None else best_validation["validation"].get("public_wrong"),
                "best_validated_arc_wrong": None if best_validation is None else best_validation["validation"].get("arc_gen_wrong"),
                "top_programs": [_candidate_quality(c) for c in enriched[:20]],
                "validations": validations,
                "top_failures": task_failures.most_common(10),
            }
            task_hits[task] = row
            rows.append(row)
        summary = {
            "tasks_checked": len(tasks),
            "tasks_with_any_python_candidate": counters["tasks_with_any_python_candidate"],
            "tasks_with_python_public_exact": counters["tasks_with_python_public_exact"],
            "tasks_with_python_public_arc_exact": counters["tasks_with_python_public_arc_exact"],
            "tasks_with_compilable_candidate": counters["tasks_with_compilable_candidate"],
            "tasks_with_cheap_compilable_candidate": counters["tasks_with_cheap_compilable_candidate"],
            "tasks_with_validated_candidate": counters["tasks_with_validated_candidate"],
            "tasks_with_accepted_candidate": counters["tasks_with_accepted_candidate"],
            "tasks_with_positive_score_gain": counters["tasks_with_positive_score_gain"],
            "family_hit_counts": dict(family_hits),
            "compiler_support_counts": dict(compiler_support),
            "top_failures_by_reason": compiler_failures.most_common(50),
            "top_validation_failures_by_reason": validation_failures.most_common(50),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(out_dir / "summary.json", summary)
        write_json(out_dir / "family_coverage.json", {"family_hit_counts": dict(family_hits), "compiler_support_counts": dict(compiler_support)})
        write_json(out_dir / "compiler_failures.json", dict(compiler_failures))
        write_json(out_dir / "task_hits.json", task_hits)
        with (out_dir / "summary.csv").open("w", newline="") as fh:
            fieldnames = [
                "task",
                "baseline_cost",
                "python_candidates",
                "python_public_exact",
                "python_public_arc_exact",
                "compilable_candidates",
                "cheap_compilable_candidates",
                "validated_candidates",
                "accepted_candidates",
                "best_validated_cost",
                "best_validated_public_wrong",
                "best_validated_arc_wrong",
            ]
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        cache.close()


if __name__ == "__main__":
    raise SystemExit(main())
