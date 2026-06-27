from __future__ import annotations

import argparse

from .config import add_common_args


def task_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--task")
    parser.add_argument("--tasks")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--top-cost", type=int)
    parser.add_argument("--min-cost", type=int)
    parser.add_argument("--families", default="")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--trials-per-task", type=int, default=1000)
    parser.add_argument("--beam-size", type=int, default=20)
    parser.add_argument("--max-combo-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-no-improve", type=int, default=1000)
    parser.add_argument("--target-gain", type=float)
    parser.add_argument("--max-models-per-task", type=int)
    parser.add_argument("--sort-by-size", action="store_true")
    parser.add_argument("--include-candidate-prev", action="store_true")
    parser.add_argument("--exclude-pattern", action="append", default=[])
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--skip-invalid-cache", action="store_true")
    add_common_args(parser)
    return parser
