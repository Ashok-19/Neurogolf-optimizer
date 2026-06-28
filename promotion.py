from __future__ import annotations

from dataclasses import dataclass
import json
import shutil
import time
from pathlib import Path
from typing import Any

from .cache import ValidationCache
from .locking import TaskLock
from .models import OptimizerConfig, ValidationResult
from .reporting import append_jsonl
from .scoring import delta_points, is_accepted
from .validate import validate_model, validation_cache_profile


@dataclass
class PromotionResult:
    promoted: bool
    accepted: bool
    reason: str | None
    validation: ValidationResult | None
    old_cost: int | None
    new_cost: int | None
    previous_candidate_backup: str | None = None
    notes_path: str | None = None
    ledger_path: str | None = None
    row: dict[str, Any] | None = None


def _rel_to_task(task_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(task_dir))
    except ValueError:
        return str(path)


def _validation_command(model_path: str, config: OptimizerConfig) -> str:
    parts = ["python3.12", "validate_task.py", "--model", model_path]
    if config.validation_mode == "full":
        parts.append("--all")
    elif getattr(config, "quick", False):
        parts.append("--quick")
    return " ".join(parts)


def acceptance_row(
    task_dir: Path,
    source_model: Path,
    promoted_model: Path,
    validation: ValidationResult,
    old_cost: int,
    config: OptimizerConfig,
    metadata: dict[str, Any] | None = None,
    previous_candidate_backup: Path | None = None,
) -> dict[str, Any]:
    task_id = task_dir.name
    new_cost = int(validation.cost or 0)
    promoted_rel = _rel_to_task(task_dir, promoted_model)
    row = {
        "task_id": task_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_model": str(source_model),
        "promoted_model": str(promoted_model),
        "promoted_model_rel": promoted_rel,
        "previous_candidate_backup": None if previous_candidate_backup is None else str(previous_candidate_backup),
        "old_cost": int(old_cost),
        "new_cost": new_cost,
        "delta_points": delta_points(int(old_cost), new_cost),
        "public_wrong": validation.public_wrong,
        "arc_gen_wrong": validation.arc_gen_wrong,
        "fresh_right": validation.fresh_right,
        "fresh_total": validation.fresh_total,
        "exact": validation.exact,
        "validation_profile": validation_cache_profile(config),
        "validation_command": _validation_command(promoted_rel, config),
        "repair_family": None,
        "repair_name": None,
        "repair_chain": None,
        "source_mutation_family": None,
    }
    meta = metadata or {}
    row.update({k: meta.get(k) for k in ("repair_family", "repair_name", "repair_chain", "source_mutation_family") if meta.get(k) is not None})
    if meta:
        row["metadata"] = meta
    return row


def write_acceptance_artifacts(task_dir: Path, row: dict[str, Any]) -> tuple[Path, Path]:
    experiments = task_dir / "experiments"
    notes_path = experiments / "optimizer_accept_notes.md"
    ledger_path = experiments / "optimizer_accept_ledger.jsonl"
    if not notes_path.exists():
        notes_path.write_text("# Optimizer Accepted Candidate Ledger\n\n")
    lines = [
        f"## {row.get('timestamp')} {row.get('task_id')}",
        "",
        f"- source model: `{row.get('source_model')}`",
        f"- promoted model: `{row.get('promoted_model')}`",
        f"- previous candidate backup: `{row.get('previous_candidate_backup')}`",
        f"- old cost: {row.get('old_cost')}",
        f"- new cost: {row.get('new_cost')}",
        f"- delta points: {row.get('delta_points')}",
        f"- public wrong: {row.get('public_wrong')}",
        f"- arc-gen wrong: {row.get('arc_gen_wrong')}",
        f"- fresh: {row.get('fresh_right')}/{row.get('fresh_total')}",
        f"- repair family: {row.get('repair_family')}",
        f"- repair name: {row.get('repair_name')}",
        f"- repair chain: `{json.dumps(row.get('repair_chain'), sort_keys=True)}`",
        f"- source mutation family: {row.get('source_mutation_family')}",
        f"- validation profile: `{row.get('validation_profile')}`",
        f"- validation command: `{row.get('validation_command')}`",
        "",
    ]
    with notes_path.open("a") as fh:
        fh.write("\n".join(lines))
    append_jsonl(ledger_path, row)
    return notes_path, ledger_path


def current_incumbent_cost(task_dir: Path, baseline_cost: int, config: OptimizerConfig, cache: ValidationCache | None) -> tuple[Path | None, int, ValidationResult | None]:
    target = task_dir / "experiments" / "candidate.onnx"
    if not target.exists():
        return None, int(baseline_cost), None
    current = validate_model(task_dir, target, config, cache)
    if is_accepted(current.to_json_dict(), int(baseline_cost), config):
        return target, int(current.cost or baseline_cost), current
    return target, int(baseline_cost), current


def promote_candidate(
    task_dir: Path,
    source_model: Path,
    config: OptimizerConfig,
    cache: ValidationCache | None,
    old_cost: int,
    *,
    baseline_cost: int | None = None,
    metadata: dict[str, Any] | None = None,
    run_best_path: Path | None = None,
) -> PromotionResult:
    if config.dry_run:
        return PromotionResult(False, False, "dry_run", None, old_cost, None)
    if not config.promote:
        return PromotionResult(False, False, "promote_disabled", None, old_cost, None)
    source_model = Path(source_model)
    if not source_model.exists():
        return PromotionResult(False, False, "source_missing", None, old_cost, None)
    baseline = int(baseline_cost if baseline_cost is not None else old_cost)
    experiments = task_dir / "experiments"
    target = experiments / "candidate.onnx"
    with TaskLock(experiments):
        fresh = validate_model(task_dir, source_model, config, cache)
        if not is_accepted(fresh.to_json_dict(), int(old_cost), config):
            return PromotionResult(False, False, "validation_not_accepted", fresh, old_cost, fresh.cost)
        _current_path, current_cost, _current_validation = current_incumbent_cost(task_dir, baseline, config, cache)
        if fresh.cost is None or int(fresh.cost) >= current_cost:
            return PromotionResult(False, True, "cost_not_below_current_incumbent", fresh, current_cost, fresh.cost)
        backup_path = None
        if target.exists():
            backup_path = experiments / f"candidate_prev_{time.strftime('%Y%m%d_%H%M%S')}.onnx"
            shutil.copy2(target, backup_path)
        shutil.copy2(source_model, target)
        if run_best_path is not None:
            shutil.copy2(source_model, run_best_path)
        row = acceptance_row(task_dir, source_model, target, fresh, current_cost, config, metadata, backup_path)
        notes_path, ledger_path = write_acceptance_artifacts(task_dir, row)
        return PromotionResult(True, True, None, fresh, current_cost, int(fresh.cost), None if backup_path is None else str(backup_path), str(notes_path), str(ledger_path), row)


def record_existing_acceptance(
    task_dir: Path,
    promoted_model: Path,
    config: OptimizerConfig,
    cache: ValidationCache | None,
    old_cost: int,
    *,
    source_model: Path | None = None,
    previous_candidate_backup: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> PromotionResult:
    promoted_model = Path(promoted_model)
    if not promoted_model.exists():
        return PromotionResult(False, False, "promoted_model_missing", None, old_cost, None)
    validation = validate_model(task_dir, promoted_model, config, cache)
    if not is_accepted(validation.to_json_dict(), int(old_cost), config):
        return PromotionResult(False, False, "validation_not_accepted", validation, old_cost, validation.cost)
    row = acceptance_row(task_dir, source_model or promoted_model, promoted_model, validation, int(old_cost), config, metadata, previous_candidate_backup)
    notes_path, ledger_path = write_acceptance_artifacts(task_dir, row)
    return PromotionResult(
        False,
        True,
        "recorded_existing_acceptance",
        validation,
        int(old_cost),
        int(validation.cost or 0),
        None if previous_candidate_backup is None else str(previous_candidate_backup),
        str(notes_path),
        str(ledger_path),
        row,
    )
