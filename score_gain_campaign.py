from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "neurogolf_optimizer"

from .cache import ValidationCache
from .cli import task_parser
from .config import config_from_args
from .models import OptimizerConfig
from .optimizer import NeuroGolfOptimizer, timestamp
from .reporting import append_jsonl, write_json
from .scoring import points
from .search.scheduler import discover_tasks


def _run_task(task_id: str, config_data: dict, run_id: str) -> dict:
    config = OptimizerConfig.from_json_dict(config_data)
    optimizer = NeuroGolfOptimizer(config, run_id=run_id)
    return optimizer.optimize_task(task_id)


def _accepted_contribution(row: dict, baseline_cost: int | None) -> dict:
    new_cost = row.get("candidate_cost")
    old_cost = None
    if new_cost is not None and row.get("cost_reduction") is not None:
        old_cost = int(new_cost + row["cost_reduction"])
    return {
        "task": row.get("task"),
        "baseline_cost": baseline_cost,
        "old_incumbent_cost": old_cost,
        "new_candidate_cost": new_cost,
        "baseline_score_contribution": None if not baseline_cost else points(baseline_cost),
        "old_candidate_score_contribution": None if old_cost is None else points(old_cost),
        "new_candidate_score_contribution": None if new_cost is None else points(new_cost),
        "delta_points": row.get("delta_points"),
        "model_path": row.get("model_path"),
        "promoted": row.get("promoted"),
        "families": row.get("families"),
        "public_wrong": row.get("public_wrong"),
        "arc_gen_wrong": row.get("arc_gen_wrong"),
        "fresh": f"{row.get('fresh_right')}/{row.get('fresh_total')}",
    }


TERMINAL_STATUSES = {"completed", "accepted", "near_miss", "timeout", "interrupted", "error", "skipped"}


def _summary_status(summary: dict) -> str:
    if summary.get("status") in TERMINAL_STATUSES:
        return str(summary["status"])
    if summary.get("error"):
        return "error"
    if summary.get("accepted"):
        return "accepted"
    if summary.get("near_misses"):
        return "near_miss"
    return "completed"


def _write_task_status(run_dir: Path, task_status: dict[str, dict]) -> None:
    write_json(run_dir / "task_status.json", task_status)


def _append_unique_row(rows: list[dict], row: dict) -> None:
    task = row.get("task")
    status = row.get("status")
    if task is not None and status is not None:
        for existing in rows:
            if existing.get("task") == task and existing.get("status") == status:
                return
    rows.append(row)


def _finalize_nonterminal_tasks(
    tasks: list[str],
    task_status: dict[str, dict],
    summaries: list[dict],
    errors: list[dict],
    *,
    running_status: str,
    pending_status: str,
    error: str,
) -> list[str]:
    finalized: list[str] = []
    for task in tasks:
        current = task_status.get(task, {"status": "pending"})
        status = current.get("status", "pending")
        if status in TERMINAL_STATUSES:
            continue
        final_status = running_status if status == "running" else pending_status
        row = {"task": task, "status": final_status, "error": error}
        if current.get("run_dir"):
            row["run_dir"] = current["run_dir"]
        task_status[task] = row
        _append_unique_row(summaries, row)
        if final_status in {"timeout", "interrupted", "error"}:
            _append_unique_row(errors, row)
        finalized.append(task)
    return finalized


def _record_progress(run_dir: Path, event: str, task: str | None, task_status: dict[str, dict], extra: dict | None = None) -> None:
    row = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "event": event,
        "task": task,
        "completed_tasks": sorted(t for t, s in task_status.items() if s.get("status") in TERMINAL_STATUSES),
        "missing_tasks": sorted(t for t, s in task_status.items() if s.get("status") not in TERMINAL_STATUSES),
    }
    if extra:
        row.update(extra)
    append_jsonl(run_dir / "progress.jsonl", row)
    _write_task_status(run_dir, task_status)


def _write_score_report(
    run_dir: Path,
    config: OptimizerConfig,
    tasks: list[str],
    summaries: list[dict],
    accepted: list[dict],
    near_misses: list[dict],
    errors: list[dict],
    current_total_score: float,
    task_status: dict[str, dict] | None = None,
) -> None:
    task_status = task_status or {}
    summary_by_task = {s.get("task"): s for s in summaries if s.get("task")}
    complete_summaries = list(summaries)
    for task in tasks:
        if task not in summary_by_task:
            status = task_status.get(task, {}).get("status", "skipped")
            complete_summaries.append({"task": task, "status": status, "error": task_status.get(task, {}).get("error")})
    baseline_by_task = {s.get("task"): s.get("baseline_cost") for s in complete_summaries}
    accepted_contrib = [_accepted_contribution(row, baseline_by_task.get(row.get("task"))) for row in accepted]
    total_gain = sum(row.get("delta_points") or 0.0 for row in accepted)
    projected_total = current_total_score + total_gain
    near_sorted = sorted(
        near_misses,
        key=lambda r: (
            0 if r.get("candidate_cost") is None else -(r.get("cost_reduction") or 0),
            -(r.get("delta_points") or 0.0),
        ),
    )
    report = {
        "tasks": tasks,
        "accepted_count": len(accepted),
        "near_miss_count": len(near_misses),
        "error_count": len(errors),
        "total_accepted_delta": total_gain,
        "current_total_score": current_total_score,
        "current_projected_total_score_if_candidates_are_used": projected_total,
        "accepted_contributions": accepted_contrib,
        "top_near_misses_by_repair_value": near_sorted[:100],
        "summaries": complete_summaries,
        "task_status": task_status,
        "completed_tasks": sorted(t for t, s in task_status.items() if s.get("status") in TERMINAL_STATUSES),
        "missing_tasks": sorted(t for t in tasks if task_status.get(t, {}).get("status") not in TERMINAL_STATUSES),
        "running_at_interrupt": sorted(t for t, s in task_status.items() if s.get("status") == "interrupted"),
        "accepted_so_far": len(accepted),
        "near_misses_so_far": len(near_misses),
    }
    write_json(run_dir / "score_gain_report.json", report)
    write_json(run_dir / "accepted.json", accepted)
    write_json(run_dir / "near_misses.json", near_misses)
    write_json(run_dir / "rejected_summary.json", {"summaries": complete_summaries, "errors": errors})
    md = [
        "# NeuroGolf Score-Gain Campaign",
        "",
        f"- tasks: {len(tasks)}",
        f"- accepted candidates: {len(accepted)}",
        f"- near misses: {len(near_misses)}",
        f"- errors: {len(errors)}",
        f"- total accepted delta: {total_gain:.6f}",
        f"- projected total score: {projected_total:.6f}",
        "",
    ]
    if accepted_contrib:
        md.extend(["## Accepted", ""])
        for row in accepted_contrib:
            md.append(
                f"- {row['task']}: {row['old_incumbent_cost']} -> {row['new_candidate_cost']} "
                f"delta={row['delta_points']} promoted={row['promoted']} `{row['model_path']}`"
            )
    if near_sorted:
        md.extend(["", "## Top Near Misses", ""])
        for row in near_sorted[:25]:
            md.append(
                f"- {row.get('task')} [{row.get('near_miss_category')}]: cost={row.get('candidate_cost')} "
                f"cut={row.get('cost_reduction')} public={row.get('public_wrong')} arc={row.get('arc_gen_wrong')} "
                f"fresh={row.get('fresh_right')}/{row.get('fresh_total')} `{row.get('model_path')}`"
            )
    (run_dir / "SCORE_GAIN_REPORT.md").write_text("\n".join(md) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser: argparse.ArgumentParser = task_parser("Run accepted-score-gain NeuroGolf campaign")
    parser.add_argument("--current-total-score", type=float, default=7157.53)
    parser.add_argument("--task-time-limit-seconds", type=int)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    args = parser.parse_args(argv)
    if args.task and not args.tasks:
        args.tasks = args.task
    config = config_from_args(args)
    config.trials = args.trials_per_task
    cache_dir = config.cache_dir or (config.repo_root / ".chatgpt/operations/neurogolf_optimizer/cache")
    if config.clear_cache:
        ValidationCache.clear(cache_dir)
    run_id = config.run_id or timestamp()
    run_dir = config.repo_root / ".chatgpt/operations/neurogolf_optimizer/score_gain_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tasks = discover_tasks(config.repo_root, all_tasks=args.all, tasks=args.tasks, top_cost=args.top_cost, min_cost=args.min_cost, preset=config.preset)
    write_json(run_dir / "config.json", config.to_json_dict() | {"current_total_score": args.current_total_score})
    write_json(run_dir / "task_order.json", tasks)
    accepted: list[dict] = []
    near_misses: list[dict] = []
    summaries: list[dict] = []
    errors: list[dict] = []
    task_status: dict[str, dict] = {task: {"status": "pending"} for task in tasks}
    status_path = run_dir / "task_status.json"
    if config.resume and status_path.exists():
        try:
            loaded = json.loads(status_path.read_text())
            if isinstance(loaded, dict):
                task_status.update({task: loaded.get(task, {"status": "pending"}) for task in tasks})
        except Exception:
            pass
    _write_task_status(run_dir, task_status)
    deadline = None if args.time_limit_seconds is None else time.time() + args.time_limit_seconds
    interrupted_signal: int | None = None

    def _handle_interrupt(signum, _frame) -> None:
        nonlocal interrupted_signal
        interrupted_signal = signum
        raise KeyboardInterrupt

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _handle_interrupt)

    def task_config_data(remaining_seconds: float | None = None) -> dict:
        data = config.to_json_dict()
        limits: list[int] = []
        if data.get("time_limit_seconds") is not None:
            limits.append(int(data["time_limit_seconds"]))
        if args.task_time_limit_seconds is not None:
            limits.append(int(args.task_time_limit_seconds))
        if remaining_seconds is not None:
            limits.append(max(1, int(remaining_seconds)))
        if limits:
            limit = max(1, min(limits))
            data["time_limit_seconds"] = limit
            data["timeout"] = min(int(data.get("timeout") or limit), limit)
        return data

    def record_summary(summary: dict) -> None:
        task = str(summary.get("task"))
        summaries.append(summary)
        accepted.extend(summary.get("accepted", []))
        near_misses.extend(summary.get("near_misses", []))
        status = _summary_status(summary)
        task_status[task] = {"status": status, "run_dir": summary.get("run_dir"), "error": summary.get("error")}
        if status == "error":
            errors.append(summary)
        _record_progress(run_dir, f"task_{status}", task, task_status)

    try:
        if not tasks:
            errors.append({"error": "no tasks selected"})
        elif config.workers <= 1:
            for task in tasks:
                if task_status.get(task, {}).get("status") in TERMINAL_STATUSES and config.resume:
                    continue
                remaining = None if deadline is None else deadline - time.time()
                if remaining is not None and remaining <= 0:
                    task_status[task] = {"status": "timeout", "error": "campaign global time limit reached before task start"}
                    err = {"task": task, "status": "timeout", "error": task_status[task]["error"]}
                    errors.append(err)
                    summaries.append(err)
                    _record_progress(run_dir, "task_timeout", task, task_status)
                    continue
                task_status[task] = {"status": "running", "started_at": time.time()}
                _record_progress(run_dir, "task_started", task, task_status)
                task_cfg = OptimizerConfig.from_json_dict(task_config_data(remaining))
                optimizer = NeuroGolfOptimizer(task_cfg, run_id=run_id)
                summary = optimizer.optimize_task(task)
                record_summary(summary)
                _write_score_report(run_dir, config, tasks, summaries, accepted, near_misses, errors, args.current_total_score, task_status)
        else:
            data = task_config_data()
            pool = ProcessPoolExecutor(max_workers=config.workers)
            pool_shutdown = False
            try:
                futures = {}
                for task in tasks:
                    if task_status.get(task, {}).get("status") in TERMINAL_STATUSES and config.resume:
                        continue
                    task_status[task] = {"status": "running", "started_at": time.time()}
                    futures[pool.submit(_run_task, task, data, run_id)] = task
                    _record_progress(run_dir, "task_started", task, task_status)
                heartbeat = max(1, int(args.heartbeat_seconds or 30))
                while futures:
                    if deadline is not None and time.time() >= deadline:
                        for fut, task in list(futures.items()):
                            status = "interrupted" if fut.running() else "timeout"
                            task_status[task] = {"status": status, "error": "campaign global time limit reached"}
                            err = {"task": task, "status": status, "error": task_status[task]["error"]}
                            errors.append(err)
                            summaries.append(err)
                        _record_progress(
                            run_dir,
                            "campaign_interrupted",
                            None,
                            task_status,
                            {
                                "accepted_so_far": len(accepted),
                                "near_misses_so_far": len(near_misses),
                                "running_at_interrupt": sorted(futures.values()),
                            },
                        )
                        _write_score_report(run_dir, config, tasks, summaries, accepted, near_misses, errors, args.current_total_score, task_status)
                        pool.shutdown(wait=False, cancel_futures=True)
                        pool_shutdown = True
                        print(
                            json.dumps(
                                {
                                    "run_dir": str(run_dir),
                                    "tasks": len(tasks),
                                    "accepted": len(accepted),
                                    "near_misses": len(near_misses),
                                    "errors": len(errors),
                                    "completed_tasks": sorted(t for t, s in task_status.items() if s.get("status") in TERMINAL_STATUSES),
                                    "missing_tasks": sorted(t for t in tasks if task_status.get(t, {}).get("status") not in TERMINAL_STATUSES),
                                    "running_at_interrupt": sorted(futures.values()),
                                    "total_accepted_delta": sum(row.get("delta_points") or 0.0 for row in accepted),
                                },
                                indent=2,
                                sort_keys=True,
                            )
                        )
                        return 124
                    done, _pending = wait(futures, timeout=heartbeat, return_when=FIRST_COMPLETED)
                    if not done:
                        _record_progress(run_dir, "heartbeat", None, task_status, {"running": sorted(futures.values())})
                        _write_score_report(run_dir, config, tasks, summaries, accepted, near_misses, errors, args.current_total_score, task_status)
                        continue
                    for fut in done:
                        task = futures.pop(fut)
                        try:
                            summary = fut.result()
                        except Exception as exc:
                            summary = {"task": task, "error": repr(exc)}
                        record_summary(summary)
                        _write_score_report(run_dir, config, tasks, summaries, accepted, near_misses, errors, args.current_total_score, task_status)
            except KeyboardInterrupt:
                _finalize_nonterminal_tasks(
                    tasks,
                    task_status,
                    summaries,
                    errors,
                    running_status="interrupted",
                    pending_status="interrupted",
                    error="campaign interrupted",
                )
                _record_progress(
                    run_dir,
                    "campaign_interrupted",
                    None,
                    task_status,
                    {"accepted_so_far": len(accepted), "near_misses_so_far": len(near_misses), "running_at_interrupt": sorted(futures.values())},
                )
                _write_score_report(run_dir, config, tasks, summaries, accepted, near_misses, errors, args.current_total_score, task_status)
                pool.shutdown(wait=False, cancel_futures=True)
                pool_shutdown = True
                return 130
            finally:
                if not pool_shutdown:
                    pool.shutdown(wait=True, cancel_futures=True)
    except KeyboardInterrupt:
        reason = "campaign interrupted" if interrupted_signal is None else f"campaign interrupted by signal {interrupted_signal}"
        finalized = _finalize_nonterminal_tasks(
            tasks,
            task_status,
            summaries,
            errors,
            running_status="interrupted",
            pending_status="interrupted",
            error=reason,
        )
        _record_progress(
            run_dir,
            "campaign_interrupted",
            None,
            task_status,
            {"accepted_so_far": len(accepted), "near_misses_so_far": len(near_misses), "running_at_interrupt": sorted(finalized)},
        )
        _write_score_report(run_dir, config, tasks, summaries, accepted, near_misses, errors, args.current_total_score, task_status)
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    if deadline is not None and time.time() >= deadline:
        finalized = _finalize_nonterminal_tasks(
            tasks,
            task_status,
            summaries,
            errors,
            running_status="interrupted",
            pending_status="timeout",
            error="campaign global time limit reached",
        )
        if finalized:
            _record_progress(
                run_dir,
                "campaign_interrupted",
                None,
                task_status,
                {"accepted_so_far": len(accepted), "near_misses_so_far": len(near_misses), "running_at_interrupt": sorted(finalized)},
            )
    _write_score_report(run_dir, config, tasks, summaries, accepted, near_misses, errors, args.current_total_score, task_status)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "tasks": len(tasks),
                "accepted": len(accepted),
                "near_misses": len(near_misses),
                "errors": len(errors),
                "total_accepted_delta": sum(row.get("delta_points") or 0.0 for row in accepted),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
