from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "neurogolf_optimizer"

from .cli import task_parser
from .config import config_from_args
from .cache import ValidationCache
from .models import OptimizerConfig
from .optimizer import NeuroGolfOptimizer, timestamp
from .reporting import write_json, write_run_report
from .search.scheduler import discover_tasks


def _run_task(task_id: str, config_data: dict, run_id: str) -> dict:
    config = OptimizerConfig.from_json_dict(config_data)
    optimizer = NeuroGolfOptimizer(config, run_id=run_id)
    return optimizer.optimize_task(task_id)


def run_campaign(argv: list[str] | None = None) -> int:
    parser = task_parser("Run NeuroGolf optimization campaign")
    args = parser.parse_args(argv)
    if args.task and not args.tasks:
        args.tasks = args.task
    config = config_from_args(args)
    config.trials = args.trials_per_task
    cache_dir = config.cache_dir or (config.repo_root / ".chatgpt/operations/neurogolf_optimizer/cache")
    if config.clear_cache:
        ValidationCache.clear(cache_dir)
    if config.cache_stats:
        cache = ValidationCache(cache_dir, disabled=config.no_cache)
        print(json.dumps(cache.stats(), indent=2, sort_keys=True))
        cache.close()
        return 0
    run_id = config.run_id or timestamp()
    run_dir = config.repo_root / ".chatgpt/operations/neurogolf_optimizer/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config.to_json_dict())
    tasks = discover_tasks(config.repo_root, all_tasks=args.all, tasks=args.tasks, top_cost=args.top_cost, min_cost=args.min_cost, preset=config.preset)
    write_json(run_dir / "tasks.json", tasks)
    accepted: list[dict] = []
    near_misses: list[dict] = []
    errors: list[dict] = []
    summaries: list[dict] = []
    if not tasks:
        errors.append({"error": "no tasks selected"})
    elif config.workers <= 1:
        optimizer = NeuroGolfOptimizer(config, run_id=run_id)
        for task in tasks:
            summary = optimizer.optimize_task(task)
            summaries.append(summary)
            accepted.extend(summary.get("accepted", []))
            near_misses.extend(summary.get("near_misses", []))
            if "error" in summary:
                errors.append(summary)
    else:
        data = config.to_json_dict()
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            futures = {pool.submit(_run_task, task, data, run_id): task for task in tasks}
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    summary = fut.result()
                except Exception as exc:
                    summary = {"task": task, "error": repr(exc)}
                summaries.append(summary)
                accepted.extend(summary.get("accepted", []))
                near_misses.extend(summary.get("near_misses", []))
                if "error" in summary:
                    errors.append(summary)
    write_json(run_dir / "summary.json", {"tasks": tasks, "summaries": summaries, "accepted": accepted, "near_misses": near_misses, "errors": errors})
    write_run_report(run_dir, accepted, near_misses, errors)
    print(json.dumps({"run_dir": str(run_dir), "tasks": len(tasks), "accepted": len(accepted), "near_misses": len(near_misses), "errors": len(errors)}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(run_campaign())
