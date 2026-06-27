from __future__ import annotations

import fnmatch
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "neurogolf_optimizer"

from .cache import ValidationCache
from .cli import task_parser
from .config import config_from_args
from .locking import TaskLock
from .models import Candidate
from .reporting import append_jsonl, candidate_summary, write_json, write_run_report, write_task_notes
from .scoring import delta_points, is_accepted, is_near_miss, near_miss_category
from .search.scheduler import discover_tasks, read_baseline_cost
from .validate import validate_model


def _model_paths(experiments: Path, args) -> list[Path]:
    if not experiments.exists():
        return []
    patterns = list(args.exclude_pattern or [])
    if not args.include_candidate_prev:
        patterns.append("candidate_prev_*.onnx")
    if not args.deep:
        patterns.append("optimizer_runs/**/*.onnx")
    paths = sorted(experiments.glob("**/*.onnx"))
    out = []
    for path in paths:
        rel = str(path.relative_to(experiments))
        if any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(path.name, p) for p in patterns):
            continue
        out.append(path)
    if args.sort_by_size:
        out.sort(key=lambda p: p.stat().st_size if p.exists() else 0)
    if args.max_models_per_task:
        out = out[: args.max_models_per_task]
    return out


def _validate_path(task_dir: Path, model_path: Path, config, cache_dir: Path):
    cache = ValidationCache(cache_dir, disabled=config.no_cache)
    try:
        return model_path, validate_model(task_dir, model_path, config, cache)
    finally:
        cache.close()


def _promote_if_better(task_dir: Path, experiments: Path, model_path: Path, cand: Candidate, baseline_cost: int, incumbent_cost: int, config) -> bool:
    if not config.promote or config.dry_run or not cand.accepted or cand.cost is None:
        return False
    target = experiments / "candidate.onnx"
    with TaskLock(experiments):
        current_cost = incumbent_cost
        cache = ValidationCache(config.cache_dir or config.repo_root / ".chatgpt/operations/neurogolf_optimizer/cache", disabled=config.no_cache)
        try:
            if target.exists():
                current = validate_model(task_dir, target, config, cache)
                if is_accepted(current.to_json_dict(), baseline_cost, config):
                    current_cost = int(current.cost or current_cost)
            fresh = validate_model(task_dir, model_path, config, cache)
        finally:
            cache.close()
        if fresh.cost is None or fresh.cost >= current_cost or not is_accepted(fresh.to_json_dict(), current_cost, config):
            return False
        if target.exists() and target.resolve() != model_path.resolve():
            shutil.copy2(target, experiments / f"candidate_prev_{time.strftime('%Y%m%d_%H%M%S')}.onnx")
        if target.resolve() != model_path.resolve():
            shutil.copy2(model_path, target)
        write_task_notes(experiments / "optimizer_accept_notes.md", cand, baseline_cost, current_cost, int(fresh.cost))
        return True


def mine(argv: list[str] | None = None) -> int:
    parser = task_parser("Mine existing task experiments for accepted candidates")
    args = parser.parse_args(argv)
    config = config_from_args(args)
    cache_dir = config.cache_dir or (config.repo_root / ".chatgpt/operations/neurogolf_optimizer/cache")
    if config.clear_cache:
        ValidationCache.clear(cache_dir)
    if config.cache_stats:
        cache = ValidationCache(cache_dir, disabled=config.no_cache)
        print(json.dumps(cache.stats(), indent=2))
        cache.close()
        return 0
    tasks = discover_tasks(config.repo_root, all_tasks=args.all, tasks=args.tasks or args.task, top_cost=args.top_cost, min_cost=args.min_cost, preset=config.preset)
    run_id = config.run_id or ("mine_" + time.strftime("%Y%m%d_%H%M%S"))
    run_dir = config.repo_root / ".chatgpt/operations/neurogolf_optimizer/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config.to_json_dict())
    write_json(run_dir / "tasks.json", tasks)
    accepted_rows: list[dict] = []
    near_rows: list[dict] = []
    errors: list[dict] = []
    try:
        for task in tasks:
            task_dir = config.repo_root / task
            experiments = task_dir / "experiments"
            baseline_cost = read_baseline_cost(task_dir)
            incumbent_cost = baseline_cost
            existing_candidate = experiments / "candidate.onnx"
            cache = ValidationCache(cache_dir, disabled=config.no_cache)
            try:
                if existing_candidate.exists():
                    result = validate_model(task_dir, existing_candidate, config, cache)
                    if is_accepted(result.to_json_dict(), baseline_cost, config):
                        incumbent_cost = int(result.cost or baseline_cost)
                        append_jsonl(run_dir / "mine_results.jsonl", {"event": "existing_candidate", "task": task, "validation": result.to_json_dict()})
            finally:
                cache.close()
            paths = _model_paths(experiments, args)
            workers = max(1, config.workers, config.validation_workers)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_validate_path, task_dir, path, config, cache_dir) for path in paths]
                for fut in as_completed(futures):
                    try:
                        model_path, result = fut.result()
                        accepted = is_accepted(result.to_json_dict(), incumbent_cost, config)
                        near = is_near_miss(result.to_json_dict(), incumbent_cost, config)
                        cand = Candidate(task, str(existing_candidate if existing_candidate.exists() else task_dir / "task" / f"{task}.onnx"), str(model_path), [], result, None, None, result.cost, None if result.cost is None else delta_points(incumbent_cost, result.cost), accepted, near, near_miss_category(result.to_json_dict(), incumbent_cost, config))
                        row = candidate_summary(cand, incumbent_cost)
                        append_jsonl(run_dir / "mine_results.jsonl", row)
                        if accepted:
                            promoted = _promote_if_better(task_dir, experiments, model_path, cand, baseline_cost, incumbent_cost, config)
                            row["promoted"] = promoted
                            accepted_rows.append(row)
                            incumbent_cost = int(result.cost or incumbent_cost)
                        elif near:
                            near_rows.append(row)
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        err = {"task": task, "error": repr(exc)}
                        errors.append(err)
                        append_jsonl(run_dir / "errors.jsonl", err)
    except KeyboardInterrupt:
        errors.append({"error": "interrupted"})
    finally:
        write_json(run_dir / "accepted.json", accepted_rows)
        write_json(run_dir / "near_misses.json", near_rows)
        write_json(run_dir / "errors.json", errors)
        write_json(run_dir / "summary.json", {"tasks": tasks, "accepted": accepted_rows, "near_misses": near_rows, "errors": errors})
        write_run_report(run_dir, accepted_rows, near_rows, errors)
    print(json.dumps({"run_dir": str(run_dir), "tasks": len(tasks), "accepted": len(accepted_rows), "near_misses": len(near_rows), "errors": len(errors)}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(mine())
