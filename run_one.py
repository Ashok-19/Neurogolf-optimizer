from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "neurogolf_optimizer"

from .cli import task_parser
from .config import config_from_args
from .cache import ValidationCache
from .optimizer import NeuroGolfOptimizer


def main(argv: list[str] | None = None) -> int:
    parser = task_parser("Optimize one NeuroGolf task")
    args = parser.parse_args(argv)
    task = args.task or (args.tasks.split(",")[0] if args.tasks else None)
    if not task:
        parser.error("--task is required")
    config = config_from_args(args)
    config.trials = args.trials
    cache_dir = config.cache_dir or (config.repo_root / ".chatgpt/operations/neurogolf_optimizer/cache")
    if config.clear_cache:
        ValidationCache.clear(cache_dir)
    if config.cache_stats:
        cache = ValidationCache(cache_dir, disabled=config.no_cache)
        print(json.dumps(cache.stats(), indent=2, sort_keys=True))
        cache.close()
        return 0
    optimizer = NeuroGolfOptimizer(config)
    summary = optimizer.optimize_task(task)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if "error" in summary else 0


if __name__ == "__main__":
    raise SystemExit(main())
