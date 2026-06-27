from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .cache import ValidationCache
from .graph_utils import model_hash
from .models import OptimizerConfig, ValidationResult


def _empty_result(task: str, model_path: str, returncode: int, stdout: str, stderr: str, error: str | None) -> ValidationResult:
    return ValidationResult(
        task=task,
        model_path=model_path,
        public_right=None,
        public_wrong=None,
        arc_gen_right=None,
        arc_gen_wrong=None,
        fresh_right=None,
        fresh_total=None,
        memory=None,
        params=None,
        cost=None,
        points=None,
        exact=False,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        error=error,
    )


def parse_validation_stdout(task: str, model_path: str, stdout: str, stderr: str = "", returncode: int = 0) -> ValidationResult:
    try:
        start = stdout.find("{")
        end = stdout.rfind("}")
        if start < 0 or end < start:
            return _empty_result(task, model_path, returncode, stdout, stderr, "validator did not emit JSON")
        data: dict[str, Any] = json.loads(stdout[start : end + 1])
        return ValidationResult(
            task=str(data.get("task", task)),
            model_path=str(data.get("model", model_path)),
            public_right=data.get("public_right"),
            public_wrong=data.get("public_wrong"),
            arc_gen_right=data.get("arc_gen_right"),
            arc_gen_wrong=data.get("arc_gen_wrong"),
            fresh_right=data.get("fresh_right"),
            fresh_total=data.get("fresh_total"),
            memory=data.get("memory"),
            params=data.get("params"),
            cost=data.get("cost"),
            points=data.get("points"),
            exact=data.get("exact"),
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            error=None if data.get("cost") is not None else "validator returned null cost",
        )
    except Exception as exc:
        return _empty_result(task, model_path, returncode, stdout, stderr, f"parse failure: {type(exc).__name__}: {exc}")


def validation_cache_profile(config: OptimizerConfig) -> str:
    mode = config.validation_mode
    command_mode = "full" if mode == "full" else "quick" if config.quick else "default"
    return "|".join(
        [
            "validator=v2",
            f"mode={mode}",
            f"command={command_mode}",
            f"quick={int(config.quick)}",
            f"timeout={int(config.timeout)}",
        ]
    )


def validate_model(
    task_dir: Path,
    model_path: Path,
    config: OptimizerConfig,
    cache: ValidationCache | None = None,
) -> ValidationResult:
    task = task_dir.name
    try:
        h = model_hash(model_path)
    except Exception as exc:
        return _empty_result(task, str(model_path), 99, "", "", f"hash failure: {type(exc).__name__}: {exc}")
    profile = validation_cache_profile(config)
    if cache is not None:
        cached = cache.get(task, h, profile)
        if cached is not None:
            return cached
    rel_model = model_path
    try:
        rel_model = model_path.relative_to(task_dir)
    except ValueError:
        pass
    cmd = ["python3.12", "validate_task.py", "--model", str(rel_model)]
    if config.validation_mode == "full":
        cmd.append("--all")
    elif config.quick:
        cmd.append("--quick")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(task_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=config.timeout,
        )
        result = parse_validation_stdout(task, str(rel_model), proc.stdout, proc.stderr, proc.returncode)
        if proc.returncode != 0 and result.error is None and not result.exact:
            result.error = "validator returned nonzero"
    except subprocess.TimeoutExpired as exc:
        result = _empty_result(task, str(rel_model), 124, exc.stdout or "", exc.stderr or "", "validator timeout")
    except Exception as exc:
        result = _empty_result(task, str(rel_model), 98, "", "", f"validator failure: {type(exc).__name__}: {exc}")
    if cache is not None:
        cache.put(task, h, result, profile)
    return result
