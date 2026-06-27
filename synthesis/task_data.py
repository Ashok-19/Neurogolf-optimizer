from __future__ import annotations

import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GridPair:
    input: list[list[int]]
    output: list[list[int]]
    source: str
    index: int


def _valid_grid(grid: Any) -> bool:
    return isinstance(grid, list) and all(isinstance(r, list) and all(isinstance(x, int) and 0 <= x <= 9 for x in r) for r in grid)


def _pair(pair: dict, source: str, index: int) -> GridPair | None:
    inp = pair.get("input")
    out = pair.get("output")
    if not _valid_grid(inp) or not _valid_grid(out):
        return None
    if len(inp) > 30 or len(out) > 30 or any(len(r) > 30 for r in inp + out):
        return None
    return GridPair(inp, out, source, index)


def load_task_json(task_dir: Path) -> dict:
    json_path = task_dir / "task" / f"{task_dir.name}.json"
    if not json_path.exists():
        return {}
    try:
        return json.loads(json_path.read_text())
    except Exception:
        return {}


def _load_pairs(task_dir: Path, sources: tuple[str, ...]) -> list[GridPair]:
    data = load_task_json(task_dir)
    pairs: list[GridPair] = []
    for source in sources:
        for i, item in enumerate(data.get(source, [])):
            gp = _pair(item, source, i)
            if gp is not None:
                pairs.append(gp)
    return pairs


def load_public_pairs(task_dir: Path) -> list[GridPair]:
    return _load_pairs(task_dir, ("train", "test"))


def load_arc_gen_pairs(task_dir: Path) -> list[GridPair]:
    return _load_pairs(task_dir, ("arc-gen", "arc_gen"))


def load_all_available_pairs(task_dir: Path) -> list[GridPair]:
    return _load_pairs(task_dir, ("train", "test", "arc-gen", "arc_gen"))


def load_generator(task_dir: Path):
    path = task_dir / "task" / "arcgen_task.py"
    if not path.exists():
        return None
    try:
        sys.path.insert(0, str((task_dir / "ARC-GEN").resolve()))
        sys.path.insert(0, str((task_dir / "ARC-GEN" / "tasks").resolve()))
        spec = importlib.util.spec_from_file_location(f"{task_dir.name}_arcgen_task", path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def _generate_one(gen):
    if hasattr(gen, "generate"):
        return gen.generate()
    for name in ("Task", "Generator"):
        cls = getattr(gen, name, None)
        if cls is not None:
            obj = cls()
            if hasattr(obj, "generate"):
                return obj.generate()
    return None


def generate_fresh_pairs(task_dir: Path, n: int, seed: int = 10000) -> list[GridPair]:
    gen = load_generator(task_dir)
    if gen is None:
        return []
    seen = {str(p.input) for p in load_all_available_pairs(task_dir)}
    out: list[GridPair] = []
    for i in range(n * 20):
        if len(out) >= n:
            break
        random.seed(seed + i)
        try:
            item = _generate_one(gen)
        except Exception:
            continue
        if not isinstance(item, dict) or str(item.get("input")) in seen:
            continue
        gp = _pair(item, "fresh_gen", len(out))
        if gp is not None:
            seen.add(str(gp.input))
            out.append(gp)
    return out


def generate_training_bank(task_dir: Path, n: int = 500, seed: int = 10000) -> list[GridPair]:
    return load_all_available_pairs(task_dir) + generate_fresh_pairs(task_dir, n=n, seed=seed)
