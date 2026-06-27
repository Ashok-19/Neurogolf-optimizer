from __future__ import annotations

import hashlib
import traceback
from typing import Protocol

import onnx

from .graph_utils import clone_model, prune_unreachable, safe_build
from .models import BuildResult, Mutation, TaskContext


class MutationFamily(Protocol):
    name: str

    def discover(self, task: TaskContext, model: onnx.ModelProto) -> list[Mutation]:
        ...

    def apply(self, task: TaskContext, model: onnx.ModelProto, mutation: Mutation) -> BuildResult:
        ...


class BaseMutationFamily:
    name = "base"

    def discover(self, task: TaskContext, model: onnx.ModelProto) -> list[Mutation]:
        return []

    def apply(self, task: TaskContext, model: onnx.ModelProto, mutation: Mutation) -> BuildResult:
        return BuildResult(model=None, ok=False, error="not implemented")

    def build(self, task: TaskContext, model: onnx.ModelProto, changed: bool = True) -> BuildResult:
        try:
            model = prune_unreachable(model)
            result = safe_build(model, allow_nonstrict_shapes=task.config.allow_nonstrict_shapes)
            result.changed = changed
            return result
        except Exception as exc:
            return BuildResult(model=None, ok=False, error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")


def mutation_id(family: str, *parts: object) -> str:
    body = "|".join(str(p) for p in parts)
    digest = hashlib.sha1(body.encode()).hexdigest()[:12]
    return f"{family}:{digest}"


def filter_family_names(families: list[BaseMutationFamily], requested: list[str]) -> list[BaseMutationFamily]:
    if not requested:
        return families
    wanted = {x.strip().lower() for x in requested}
    return [f for f in families if f.name.lower() in wanted or f.name.split("_")[0].lower() in wanted]
