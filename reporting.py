from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Iterable

from .models import Candidate, OptimizerConfig
from .scoring import delta_points


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def candidate_summary(candidate: Candidate, incumbent_cost: int | None = None) -> dict:
    v = candidate.validation
    old = incumbent_cost or 0
    cost = candidate.cost
    return {
        "task": candidate.task,
        "model_path": candidate.model_path,
        "parent_model": candidate.parent_model,
        "candidate_cost": cost,
        "cost_reduction": None if cost is None or old <= 0 else old - cost,
        "delta_points": None if cost is None or old <= 0 else delta_points(old, cost),
        "public_wrong": None if v is None else v.public_wrong,
        "arc_gen_wrong": None if v is None else v.arc_gen_wrong,
        "fresh_right": None if v is None else v.fresh_right,
        "fresh_total": None if v is None else v.fresh_total,
        "families": [m.family for m in candidate.mutations],
        "mutations": [m.to_json_dict() for m in candidate.mutations],
        "accepted": candidate.accepted,
        "near_miss": candidate.near_miss,
        "near_miss_category": candidate.near_miss_category,
        "build_error": candidate.build_error,
        "model_hash": candidate.model_hash,
        "parent_hash": candidate.parent_hash,
        "depth": candidate.depth,
        "estimated_savings": candidate.estimated_savings,
        "promoted": candidate.promoted,
        "model_file_exists": bool(candidate.model_path and Path(candidate.model_path).exists()) if candidate.model_file_exists is None else candidate.model_file_exists,
        "retention_reason": candidate.retention_reason,
    }


def write_task_notes(path: Path, candidate: Candidate, baseline_cost: int | None, old_cost: int, new_cost: int) -> None:
    lines = [
        f"# Optimizer acceptance {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- baseline cost: {baseline_cost}",
        f"- old incumbent cost: {old_cost}",
        f"- new cost: {new_cost}",
        f"- delta points: {delta_points(old_cost, new_cost):.6f}",
        f"- model path: `{candidate.model_path}`",
        "",
        "## Validation",
        "",
    ]
    if candidate.validation:
        v = candidate.validation
        lines.extend(
            [
                f"- public wrong: {v.public_wrong}",
                f"- arc-gen wrong: {v.arc_gen_wrong}",
                f"- fresh: {v.fresh_right}/{v.fresh_total}",
                f"- exact: {v.exact}",
            ]
        )
    lines.extend(["", "## Mutations", ""])
    for m in candidate.mutations:
        lines.append(f"- {m.family}: {m.description} `{json.dumps(m.metadata, sort_keys=True)}`")
    path.write_text("\n".join(lines) + "\n")


def write_run_report(run_dir: Path, accepted: list[dict], near_misses: list[dict], errors: list[dict]) -> None:
    total_gain = sum(x.get("delta_points") or 0.0 for x in accepted)
    md = ["# NeuroGolf Optimizer Report", "", f"- accepted candidates: {len(accepted)}", f"- near misses: {len(near_misses)}", f"- errors: {len(errors)}", f"- total accepted score gain: {total_gain:.6f}", ""]
    if accepted:
        md.extend(["## Accepted", ""])
        for row in accepted:
            md.append(
                f"- {row.get('task')}: {row.get('candidate_cost')} "
                f"gain={row.get('delta_points')} fresh={row.get('fresh_right')}/{row.get('fresh_total')} `{row.get('model_path')}`"
            )
    if near_misses:
        md.extend(["", "## Near Misses", ""])
        for row in near_misses[:100]:
            md.append(
                f"- {row.get('task')} [{row.get('near_miss_category')}]: cost={row.get('candidate_cost')} "
                f"public_wrong={row.get('public_wrong')} arc_wrong={row.get('arc_gen_wrong')} fresh={row.get('fresh_right')}/{row.get('fresh_total')}"
            )
    (run_dir / "REPORT.md").write_text("\n".join(md) + "\n")
    write_json(run_dir / "campaign_summary.json", {"accepted": accepted, "near_misses": near_misses, "errors": errors, "total_gain": total_gain})
    write_json(run_dir / "accepted.json", accepted)
    write_json(run_dir / "near_misses.json", near_misses)
    write_json(run_dir / "errors.json", errors)
    with (run_dir / "summary.csv").open("w", newline="") as fh:
        fieldnames = ["task", "candidate_cost", "cost_reduction", "delta_points", "public_wrong", "arc_gen_wrong", "fresh_right", "fresh_total", "model_path", "accepted", "near_miss_category", "promoted", "depth"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in accepted + near_misses:
            writer.writerow(row)
