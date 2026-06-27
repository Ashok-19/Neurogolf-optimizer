from __future__ import annotations

import json
import math
from pathlib import Path

from ..analysis import analyze_model

PRESETS = {
    "high-cost": "task233,task018,task209,task286,task219,task118,task205,task349,task366,task191".split(","),
    "where-heavy": "task233,task018,task209,task286,task219,task364,task080,task396,task054,task023,task319,task101,task157,task096,task131,task093".split(","),
    "table-heavy": "task394,task096,task324,task370,task328,task107,task264,task280".split(","),
    "dtype-heavy": "task118,task029,task110,task202,task255,task018,task349,task055,task222,task025,task174,task066,task191".split(","),
    "conv-heavy": "task118,task029,task110,task202,task255,task018,task349,task055,task222,task025,task174,task066,task191".split(","),
    "known-good": "task014,task055,task066,task092,task118,task165,task233,task255,task286,task366,task396".split(","),
}


def read_baseline_cost(task_dir: Path) -> int:
    stats_path = task_dir / "baseline_stats.json"
    if not stats_path.exists():
        return 0
    try:
        stats = json.loads(stats_path.read_text())
    except Exception:
        return 0
    for key in ("cost", "baseline_cost", "memory_params", "total_cost"):
        if isinstance(stats.get(key), (int, float)):
            return int(stats[key])
    memory = stats.get("memory")
    params = stats.get("params")
    if isinstance(memory, (int, float)) and isinstance(params, (int, float)):
        return int(memory + params)
    return 0


def task_priority(task_dir: Path) -> tuple:
    task_id = task_dir.name
    cost = read_baseline_cost(task_dir)
    try:
        analysis = analyze_model(task_id, task_dir / "task" / f"{task_id}.onnx")
        where_bonus = analysis.where_count
        complexity = analysis.node_count
        large = len(analysis.large_lookup_table_candidates)
        float_bytes = analysis.estimated_float32_intermediate_bytes
    except Exception:
        where_bonus = complexity = large = float_bytes = 0
    cut_needed = cost - cost / math.e if cost else 0
    return (cost, where_bonus, large, complexity, float_bytes, cut_needed)


def discover_tasks(
    repo_root: Path,
    all_tasks: bool = False,
    tasks: str | None = None,
    top_cost: int | None = None,
    min_cost: int | None = None,
    preset: str | None = None,
) -> list[str]:
    if preset == "all":
        all_tasks = True
    if tasks:
        out = [t.strip() for t in tasks.split(",") if t.strip()]
    elif preset in PRESETS:
        out = PRESETS[preset][:]
    elif all_tasks:
        out = sorted(p.name for p in repo_root.glob("task[0-9][0-9][0-9]") if (p / "validate_task.py").exists())
    else:
        out = []
    if min_cost is not None:
        out = [t for t in out if read_baseline_cost(repo_root / t) >= min_cost]
    out.sort(key=lambda t: task_priority(repo_root / t), reverse=True)
    if top_cost is not None:
        out = out[:top_cost]
    return out
