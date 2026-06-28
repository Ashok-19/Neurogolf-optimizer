from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "neurogolf_optimizer"

from .cache import ValidationCache
from .config import add_common_args, config_from_args
from .near_miss_lab import try_repair_model
from .pair_runner import run_pairs
from .promotion import current_incumbent_cost, promote_candidate
from .repair_classification import classify_diagnostics
from .reporting import append_jsonl, write_json
from .scoring import delta_points, fresh_rate, is_accepted
from .search.scheduler import discover_tasks, read_baseline_cost


SUPPORTED_FAILURES = {
    "missing_last_column",
    "missing_column",
    "missing_bottom_row",
    "missing_row",
    "missing_border",
}


@dataclass
class ReplayCandidate:
    task_id: str
    model_path: str
    candidate_cost: int | None
    incumbent_cost: int | None
    public_wrong: int | None
    arc_gen_wrong: int | None
    fresh_right: int | None
    fresh_total: int | None
    near_miss_category: str | None
    source_run: str
    source_mutations: list[dict]
    source_mutation_families: list[str]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def _task_from_path(path: Path) -> str | None:
    for part in path.parts:
        if len(part) == 7 and part.startswith("task") and part[4:].isdigit():
            return part
    return None


def _flatten_rows(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, dict):
        if "model_path" in data or "validation" in data or "model" in data:
            yield data
            return
        for value in data.values():
            yield from _flatten_rows(value)
    elif isinstance(data, list):
        for item in data:
            yield from _flatten_rows(item)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return list(_flatten_rows(json.loads(path.read_text() or "[]")))
    except Exception:
        return []


def _resolve_model_path(root: Path, task_id: str, model_path: str | None) -> Path | None:
    if not model_path:
        return None
    raw = Path(model_path)
    candidates = [raw] if raw.is_absolute() else [root / raw, root / task_id / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _fresh_from_row(row: dict[str, Any]) -> tuple[int | None, int | None]:
    validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
    return _int_or_none(row.get("fresh_right", validation.get("fresh_right"))), _int_or_none(row.get("fresh_total", validation.get("fresh_total")))


def _normalize_near_miss_row(root: Path, source_path: Path, row: dict[str, Any]) -> ReplayCandidate | None:
    task_id = str(row.get("task") or row.get("task_id") or _task_from_path(source_path) or "")
    if not task_id:
        return None
    validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
    model_value = row.get("model_path") or row.get("model") or validation.get("model_path")
    resolved = _resolve_model_path(root, task_id, str(model_value) if model_value else None)
    if resolved is None:
        return None
    mutations = row.get("mutations")
    if not isinstance(mutations, list):
        mutations = row.get("source_mutations") if isinstance(row.get("source_mutations"), list) else []
    families = row.get("source_mutation_families")
    if not isinstance(families, list):
        families = row.get("families") if isinstance(row.get("families"), list) else []
    if not families and mutations:
        families = [m.get("family") for m in mutations if isinstance(m, dict) and m.get("family")]
    fresh_right, fresh_total = _fresh_from_row(row)
    return ReplayCandidate(
        task_id=task_id,
        model_path=str(resolved),
        candidate_cost=_int_or_none(row.get("candidate_cost", validation.get("cost"))),
        incumbent_cost=_int_or_none(row.get("incumbent_cost")),
        public_wrong=_int_or_none(row.get("public_wrong", validation.get("public_wrong"))),
        arc_gen_wrong=_int_or_none(row.get("arc_gen_wrong", validation.get("arc_gen_wrong"))),
        fresh_right=fresh_right,
        fresh_total=fresh_total,
        near_miss_category=row.get("near_miss_category"),
        source_run=str(source_path),
        source_mutations=[dict(m) for m in mutations if isinstance(m, dict)],
        source_mutation_families=[str(f) for f in families if f],
    )


def _near_miss_lab_candidate(root: Path, path: Path) -> ReplayCandidate | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    validation = data.get("validation") if isinstance(data.get("validation"), dict) else {}
    row = {
        "task": data.get("task") or _task_from_path(path),
        "model": data.get("model"),
        "candidate_cost": validation.get("cost"),
        "incumbent_cost": data.get("incumbent_cost"),
        "public_wrong": validation.get("public_wrong"),
        "arc_gen_wrong": validation.get("arc_gen_wrong"),
        "fresh_right": validation.get("fresh_right"),
        "fresh_total": validation.get("fresh_total"),
        "near_miss_category": data.get("near_miss_category"),
    }
    return _normalize_near_miss_row(root, path, row)


def scan_near_misses(root: Path, tasks: list[str] | None = None) -> list[ReplayCandidate]:
    selected_tasks = tasks or sorted(p.name for p in root.glob("task[0-9][0-9][0-9]") if (p / "experiments").exists())
    out: list[ReplayCandidate] = []
    seen: set[tuple[str, str]] = set()
    for task_id in selected_tasks:
        task_dir = root / task_id
        artifacts = list((task_dir / "experiments" / "optimizer_runs").glob("*/near_misses.json"))
        artifacts += list((task_dir / "experiments" / "optimizer_runs").glob("*/near_misses.jsonl"))
        for artifact in artifacts:
            for row in _load_rows(artifact):
                candidate = _normalize_near_miss_row(root, artifact, row)
                if candidate is None:
                    continue
                key = (candidate.task_id, candidate.model_path)
                if key not in seen:
                    seen.add(key)
                    out.append(candidate)
        for artifact in (task_dir / "experiments" / "near_miss_lab").glob("*/near_miss_lab.json"):
            candidate = _near_miss_lab_candidate(root, artifact)
            if candidate is None:
                continue
            key = (candidate.task_id, candidate.model_path)
            if key not in seen:
                seen.add(key)
                out.append(candidate)
    return out


def _candidate_priority(candidate: ReplayCandidate, fresh_min: float) -> tuple:
    category_rank = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}.get(candidate.near_miss_category or "", 9)
    public_exact = candidate.public_wrong == 0
    low_arc = candidate.arc_gen_wrong is not None and candidate.arc_gen_wrong <= 5
    fresh_ok = fresh_rate({"fresh_right": candidate.fresh_right, "fresh_total": candidate.fresh_total}) >= fresh_min
    below_incumbent = candidate.candidate_cost is not None and candidate.incumbent_cost is not None and candidate.candidate_cost < candidate.incumbent_cost
    possible_delta = 0 if candidate.candidate_cost is None or candidate.incumbent_cost is None else candidate.incumbent_cost - candidate.candidate_cost
    risk = 0
    if not public_exact:
        risk += 10
    if not low_arc:
        risk += 5
    if not fresh_ok:
        risk += 3
    if not below_incumbent:
        risk += 2
    return (category_rank, risk, -possible_delta, candidate.task_id, candidate.model_path)


def prioritize_near_misses(candidates: list[ReplayCandidate], fresh_min: float) -> list[ReplayCandidate]:
    return sorted(candidates, key=lambda c: _candidate_priority(c, fresh_min))


def _task_list(root: Path, task_values: list[str] | None, tasks_csv: str | None, max_tasks: int | None) -> list[str]:
    pieces: list[str] = []
    for value in task_values or []:
        pieces.extend(t.strip() for t in value.split(",") if t.strip())
    if tasks_csv:
        pieces.extend(t.strip() for t in tasks_csv.split(",") if t.strip())
    if not pieces:
        pieces = discover_tasks(root, all_tasks=True)
    out = list(dict.fromkeys(pieces))
    return out[:max_tasks] if max_tasks is not None else out


def _report_markdown(summary: dict[str, Any], per_task_best: dict[str, dict]) -> str:
    lines = [
        "# Repair Replay Campaign",
        "",
        f"- tasks scanned: {summary['tasks_scanned']}",
        f"- near misses found: {summary['near_misses_found']}",
        f"- near misses selected: {summary['near_misses_selected']}",
        f"- supported failure classes: {summary['supported_failure_classes_count']}",
        f"- repairs attempted: {summary['repairs_attempted']}",
        f"- accepted count: {summary['accepted_count']}",
        f"- promoted count: {summary['promoted_count']}",
        f"- total delta points: {summary['total_delta_points']}",
        "",
        "## Per-Task Best",
        "",
    ]
    for task, row in sorted(per_task_best.items()):
        lines.append(f"- {task}: accepted={row.get('accepted')} promoted={row.get('promoted')} cost={row.get('cost')} repair={row.get('repair')}")
    if summary.get("remaining_blockers"):
        lines.extend(["", "## Remaining Blockers", ""])
        for blocker, count in sorted(summary["remaining_blockers"].items()):
            lines.append(f"- {blocker}: {count}")
    return "\n".join(lines) + "\n"


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    config = config_from_args(args)
    config.dry_run = bool(args.dry_run or not args.promote)
    config.promote = bool(args.promote)
    root = config.repo_root
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = root / ".chatgpt/operations/neurogolf_optimizer/repair_replay_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = ValidationCache(config.cache_dir or (root / ".chatgpt/operations/neurogolf_optimizer/cache"), disabled=config.no_cache)
    tasks = _task_list(root, args.task, args.tasks, args.max_tasks)
    accepted: list[dict] = []
    rejected: list[dict] = []
    skipped: list[dict] = []
    errors: list[dict] = []
    per_task_best: dict[str, dict] = {}
    supported_failure_classes_count = 0
    repairs_attempted = 0
    promoted_count = 0
    try:
        found = scan_near_misses(root, tasks)
        selected = prioritize_near_misses(found, config.fresh_min)[: max(0, args.max_candidates)]
        for idx, candidate in enumerate(selected):
            task_dir = root / candidate.task_id
            candidate_dir = run_dir / candidate.task_id / f"candidate_{idx:04d}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            baseline_cost = read_baseline_cost(task_dir)
            _inc_path, current_cost, _inc_validation = current_incumbent_cost(task_dir, baseline_cost, config, cache)
            try:
                diagnostics = run_pairs(task_dir, Path(candidate.model_path), sources="public,arc", comparison="official_public_arc")
                classes = classify_diagnostics(diagnostics)
                class_names = [cls.kind for cls in classes]
                supported = [kind for kind in class_names if kind in SUPPORTED_FAILURES]
                if supported:
                    supported_failure_classes_count += 1
                if args.only_supported_failures and not supported:
                    row = candidate.to_json_dict() | {"skip_reason": "unsupported_failure_class", "failure_classes": class_names}
                    skipped.append(row)
                    append_jsonl(run_dir / "skipped.jsonl", row)
                    continue
                repairs = try_repair_model(
                    task_dir,
                    Path(candidate.model_path),
                    candidate_dir,
                    config,
                    cache,
                    current_cost,
                    diagnostics=diagnostics,
                    max_repairs=config.max_near_miss_repairs,
                    source_provenance=candidate.source_mutations,
                )
                repairs_attempted += len(repairs)
                for repair in repairs:
                    row = {
                        "task_id": candidate.task_id,
                        "candidate": candidate.to_json_dict(),
                        "current_incumbent_cost": current_cost,
                        "failure_classes": class_names,
                        "repair": repair,
                    }
                    validation = repair.get("validation") if isinstance(repair.get("validation"), dict) else {}
                    accepted_now = bool(repair.get("accepted") and is_accepted(validation, current_cost, config))
                    if accepted_now:
                        row["accepted"] = True
                        row["cost"] = validation.get("cost")
                        row["delta_points"] = None if validation.get("cost") is None else delta_points(current_cost, int(validation["cost"]))
                        promoted = False
                        if config.promote and not config.dry_run:
                            result = promote_candidate(
                                task_dir,
                                Path(repair["model_path"]),
                                config,
                                cache,
                                current_cost,
                                baseline_cost=baseline_cost,
                                metadata={
                                    "repair_family": repair.get("repair_family"),
                                    "repair_name": repair.get("repair"),
                                    "repair_chain": repair.get("repair_chain"),
                                    "source_mutation_family": repair.get("source_mutation_family"),
                                    "repair_replay_source": candidate.to_json_dict(),
                                },
                            )
                            promoted = result.promoted
                            row["promotion_result"] = None if result.row is None else result.row
                        row["promoted"] = promoted
                        promoted_count += 1 if promoted else 0
                        accepted.append(row)
                        append_jsonl(run_dir / "accepted.jsonl", row)
                        best = per_task_best.get(candidate.task_id)
                        if best is None or (row.get("cost") or 10**18) < (best.get("cost") or 10**18):
                            per_task_best[candidate.task_id] = {"accepted": True, "promoted": promoted, "cost": row.get("cost"), "repair": repair.get("repair")}
                    else:
                        reason = repair.get("skip_reason") or (repair.get("cost_gate") or {}).get("reason") or repair.get("error") or "not_accepted"
                        row["accepted"] = False
                        row["promoted"] = False
                        row["skip_reason"] = reason
                        rejected.append(row)
                        append_jsonl(run_dir / "rejected.jsonl", row)
                        per_task_best.setdefault(candidate.task_id, {"accepted": False, "promoted": False, "cost": validation.get("cost"), "repair": repair.get("repair"), "reason": reason})
            except Exception as exc:
                row = candidate.to_json_dict() | {"error": f"{type(exc).__name__}: {exc}"}
                errors.append(row)
                append_jsonl(run_dir / "errors.jsonl", row)
        blockers: dict[str, int] = {}
        for row in skipped + rejected:
            reason = str(row.get("skip_reason") or "unknown")
            blockers[reason] = blockers.get(reason, 0) + 1
        summary = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "dry_run": config.dry_run,
            "tasks_scanned": len(tasks),
            "near_misses_found": len(found),
            "near_misses_selected": len(selected),
            "supported_failure_classes_count": supported_failure_classes_count,
            "repairs_attempted": repairs_attempted,
            "accepted_count": len(accepted),
            "promoted_count": promoted_count,
            "skipped_count": len(skipped),
            "rejected_count": len(rejected),
            "error_count": len(errors),
            "total_delta_points": sum(row.get("delta_points") or 0.0 for row in accepted),
            "per_task_best": per_task_best,
            "remaining_blockers": blockers,
        }
        report = {"summary": summary, "accepted": accepted, "skipped": skipped, "rejected": rejected, "errors": errors}
        write_json(run_dir / "repair_replay_report.json", report)
        (run_dir / "repair_replay_report.md").write_text(_report_markdown(summary, per_task_best))
        return report
    finally:
        cache.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay retained NeuroGolf near misses through internal repair")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--tasks")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--only-supported-failures", action="store_true", default=True)
    parser.add_argument("--include-unsupported-failures", action="store_false", dest="only_supported_failures")
    add_common_args(parser)
    args = parser.parse_args(argv)
    report = run_replay(args)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
