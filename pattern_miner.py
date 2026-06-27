from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "neurogolf_optimizer"

from .graph_utils import (
    get_consumers,
    get_producers,
    get_tensor_elem_type,
    get_value_info_shapes,
    initializer_arrays,
    load_model,
    tensor_dtype_size,
)
from .models import OptimizerConfig, TaskContext
from .mutations import default_families
from .mutations.broadcast_scalarize import BROADCAST_SAFE_OPS, can_scalarize_initializer_for_node
from .reporting import write_json
from .search.scheduler import discover_tasks, read_baseline_cost


@dataclass
class PatternHit:
    pattern_id: str
    task_id: str
    model_path: str
    estimated_savings: int
    rewrite_family: str
    ort_legality: str
    static_shape_confidence: float
    risk: float
    operators: list[str] = field(default_factory=list)
    dtypes: list[str] = field(default_factory=list)
    example_nodes: list[str] = field(default_factory=list)
    example_initializers: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    test_fixture: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternOpportunity:
    pattern_id: str
    affected_task_count: int
    affected_tasks: list[str]
    estimated_savings: int
    rewrite_family: str
    ort_legality: str
    static_shape_confidence: float
    risk: float
    rank_score: float
    sentinel_task: str
    operators: list[str]
    dtypes: list[str]
    example_nodes: list[str]
    example_initializers: list[str]
    constraints: list[str]
    test_fixture: str
    example_hits: list[dict[str, Any]]


def accepted_incumbent_path(task_dir: Path) -> tuple[Path, str]:
    task_id = task_dir.name
    candidate = task_dir / "experiments" / "candidate.onnx"
    notes = task_dir / "experiments" / "optimizer_accept_notes.md"
    if candidate.exists() and notes.exists():
        return candidate, "accepted_candidate"
    return task_dir / "task" / f"{task_id}.onnx", "baseline"


def _dtype_name(elem_type: int | None) -> str:
    if elem_type is None:
        return "UNKNOWN"
    try:
        return TensorProto.DataType.Name(elem_type)
    except ValueError:
        return str(elem_type)


def _node_label(node: onnx.NodeProto, idx: int) -> str:
    name = node.name or ",".join(node.output[:1]) or f"node_{idx}"
    return f"{idx}:{node.op_type}:{name}"


def _shape_numel(shape: list[int | None] | None) -> int | None:
    if not shape or any(dim is None for dim in shape):
        return None
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _hit(
    pattern_id: str,
    task_id: str,
    model_path: Path,
    estimated_savings: int,
    rewrite_family: str,
    ort_legality: str,
    static_shape_confidence: float,
    risk: float,
    **kwargs: Any,
) -> PatternHit:
    return PatternHit(
        pattern_id=pattern_id,
        task_id=task_id,
        model_path=str(model_path),
        estimated_savings=max(0, int(estimated_savings)),
        rewrite_family=rewrite_family,
        ort_legality=ort_legality,
        static_shape_confidence=static_shape_confidence,
        risk=risk,
        **kwargs,
    )


def _family_hits(task_id: str, model_path: Path, model: onnx.ModelProto, ctx: TaskContext) -> list[PatternHit]:
    hits: list[PatternHit] = []
    for family in default_families():
        try:
            mutations = family.discover(ctx, model)
        except Exception as exc:
            hits.append(
                _hit(
                    f"family_discover_error:{family.name}",
                    task_id,
                    model_path,
                    0,
                    family.name,
                    "unknown",
                    0.0,
                    1.0,
                    metadata={"error": repr(exc)},
                )
            )
            continue
        if not mutations:
            continue
        by_kind: dict[str, list] = defaultdict(list)
        for mutation in mutations:
            kind = str(mutation.metadata.get("kind") or family.name)
            by_kind[kind].append(mutation)
        for kind, group in by_kind.items():
            estimated = sum(m.estimated_savings or 0 for m in group)
            example_nodes = []
            for mutation in group[:5]:
                node_index = mutation.metadata.get("node_index")
                if isinstance(node_index, int):
                    example_nodes.append(f"{node_index}:{mutation.description}")
                else:
                    example_nodes.append(mutation.description)
            hits.append(
                _hit(
                    f"mutation_family:{family.name}:{kind}",
                    task_id,
                    model_path,
                    estimated,
                    family.name,
                    _family_ort_legality(family.name, kind),
                    _family_shape_confidence(family.name, kind),
                    _family_risk(family.name, kind),
                    example_nodes=example_nodes[:5],
                    constraints=_family_constraints(family.name, kind),
                    test_fixture=f"tests/test_mutation_generation.py::{family.name}",
                    metadata={"mutation_count": len(group), "top_mutations": [m.to_json_dict() for m in group[:5]]},
                )
            )
    return hits


def _family_ort_legality(family: str, kind: str) -> str:
    if family in {"concat_pad", "broadcast_scalarize", "gather_shrink", "dtype_lowering", "scalar_label_rewrite", "where_simplify"}:
        return "local_shape_checked"
    if family in {"where_ablation", "where_combo", "output_collapse", "bool_mask_rewrite"}:
        return "operator_supported_behavior_risky"
    if "error" in kind:
        return "unknown"
    return "safe_build_checked"


def _family_shape_confidence(family: str, kind: str) -> float:
    if family in {"concat_pad", "broadcast_scalarize", "initializer_prune", "initializer_dedup", "gather_shrink", "where_simplify"}:
        return 0.95
    if family in {"dtype_lowering", "scalar_label_rewrite", "cast_simplify", "constant_rewrite"}:
        return 0.90
    if family in {"dimension_sweep", "where_ablation", "where_combo", "output_collapse", "bool_mask_rewrite"}:
        return 0.70
    if "error" in kind:
        return 0.0
    return 0.80


def _family_risk(family: str, kind: str) -> float:
    if family in {"initializer_prune", "initializer_dedup", "broadcast_scalarize", "constant_rewrite", "where_simplify"}:
        return 0.05
    if family in {"concat_pad", "gather_shrink", "dtype_lowering", "scalar_label_rewrite", "cast_simplify"}:
        return 0.20
    if family in {"dimension_sweep", "where_ablation", "where_combo", "output_collapse", "bool_mask_rewrite"}:
        return 0.65
    if "error" in kind:
        return 1.0
    return 0.35


def _family_constraints(family: str, kind: str) -> list[str]:
    if family == "concat_pad":
        return ["zero initializer input", "Pad dtype accepted by ORT", "static concat rank"]
    if family == "broadcast_scalarize":
        return ["uniform initializer", "all consumers support multidirectional broadcasting", "non-scalar shape anchor remains"]
    if family == "gather_shrink":
        return ["constant indices", "used table slices are a strict subset", "indices remap in range"]
    if family == "dtype_lowering":
        return ["consumer operators accept lowered dtype", "numeric values exactly representable"]
    if family in {"where_ablation", "where_combo", "output_collapse"}:
        return ["official validation required", "branch may be data-dependent"]
    if family == "where_simplify":
        return ["identical branches or static all-true/all-false condition", "shape inference must preserve output shape"]
    if family == "dimension_sweep":
        return ["static arange/work dimension", "official validation required"]
    return ["safe_build must pass"]


def _concat_hits(task_id: str, model_path: Path, model: onnx.ModelProto) -> list[PatternHit]:
    arrays = initializer_arrays(model)
    hits: list[PatternHit] = []
    for idx, node in enumerate(model.graph.node):
        if node.op_type != "Concat":
            continue
        axis = 0
        for attr in node.attribute:
            if attr.name == "axis":
                axis = int(attr.i)
        zero_inputs = []
        for inp in node.input:
            arr = arrays.get(inp)
            if arr is not None and arr.size and np.all(arr == 0):
                zero_inputs.append((inp, arr))
        if not zero_inputs:
            continue
        dtype = str(zero_inputs[0][1].dtype)
        suffix = "tail" if node.input[-1] == zero_inputs[0][0] else "head" if node.input[0] == zero_inputs[0][0] else "middle"
        hits.append(
            _hit(
                f"concat_zero_{suffix}_{dtype}",
                task_id,
                model_path,
                sum(int(arr.size) for _name, arr in zero_inputs),
                "concat_pad",
                "verified_except_bool_pad" if dtype == "bool" else "verified_by_concat_pad",
                0.95 if dtype != "bool" else 0.60,
                0.20 if dtype != "bool" else 0.45,
                operators=["Concat", "Pad"],
                dtypes=[dtype],
                example_nodes=[_node_label(node, idx)],
                example_initializers=[name for name, _arr in zero_inputs[:5]],
                constraints=[
                    "zero initializer input",
                    "Pad replacement only for two-input non-bool concats",
                    "reuse-only fallback for duplicate zero tensors",
                ],
                test_fixture="tests/test_mutation_generation.py::test_concat_pad_rewrite_uses_zero_initializer_dtype",
                metadata={"axis": axis, "input_count": len(node.input), "zero_input_count": len(zero_inputs), "output": list(node.output)},
            )
        )
    return hits


def _initializer_hits(task_id: str, model_path: Path, model: onnx.ModelProto) -> list[PatternHit]:
    hits: list[PatternHit] = []
    arrays = initializer_arrays(model)
    consumers = get_consumers(model)
    shapes = get_value_info_shapes(model)
    signatures: dict[tuple[int, tuple[int, ...], bytes], list[str]] = defaultdict(list)
    for init in model.graph.initializer:
        arr = arrays.get(init.name)
        if arr is None:
            continue
        signatures[(int(init.data_type), tuple(init.dims), arr.tobytes())].append(init.name)
        dtype = str(arr.dtype)
        elem_size = int(arr.dtype.itemsize) if arr.dtype.itemsize else tensor_dtype_size(int(init.data_type))
        consumer_ops = sorted({node.op_type for node in consumers.get(init.name, [])})
        if arr.size >= 128 and np.all(arr == 0):
            hits.append(
                _hit(
                    f"large_zero_initializer_{dtype}",
                    task_id,
                    model_path,
                    int(arr.size * elem_size),
                    _zero_initializer_rewrite_family(consumer_ops),
                    "consumer_dependent",
                    0.75,
                    0.35,
                    operators=consumer_ops,
                    dtypes=[dtype],
                    example_initializers=[init.name],
                    constraints=["all-zero payload", "rewrite depends on consumer op broadcast/shape rules"],
                    test_fixture="tests/test_mutation_generation.py::test_broadcast_scalarize_rewrites_uniform_where_branch",
                    metadata={"shape": list(arr.shape), "consumers": consumer_ops},
                )
            )
        if (
            arr.size >= 16
            and consumer_ops
            and set(consumer_ops) <= BROADCAST_SAFE_OPS
            and np.all(arr.reshape(-1) == arr.reshape(-1)[0])
            and all(can_scalarize_initializer_for_node(node, init.name, list(arr.shape), shapes) for node in consumers.get(init.name, []))
        ):
            hits.append(
                _hit(
                    f"broadcast_uniform_scalar_{dtype}",
                    task_id,
                    model_path,
                    int(max(0, arr.size - 1) * elem_size),
                    "broadcast_scalarize",
                    "broadcast_ops_supported",
                    0.95,
                    0.05,
                    operators=consumer_ops,
                    dtypes=[dtype],
                    example_initializers=[init.name],
                    constraints=["uniform initializer", "broadcast-safe consumers only", "shape anchor required by family"],
                    test_fixture="tests/test_mutation_generation.py::test_broadcast_scalarize_rewrites_uniform_where_branch",
                    metadata={"shape": list(arr.shape), "value": _json_scalar(arr.reshape(-1)[0]), "consumers": consumer_ops},
                )
            )
        if arr.size >= 512 and arr.ndim >= 2:
            flat = arr.reshape(-1, arr.shape[-1])
            try:
                unique_ratio = float(len(np.unique(flat, axis=0))) / max(1, flat.shape[0])
            except Exception:
                unique_ratio = 1.0
            if unique_ratio < 0.75:
                hits.append(
                    _hit(
                        f"large_lookup_table_{dtype}",
                        task_id,
                        model_path,
                        int(arr.size * elem_size * (1.0 - unique_ratio)),
                        "gather_shrink|table_shrink|rule_compiler",
                        "consumer_dependent",
                        0.80,
                        0.30,
                        operators=consumer_ops,
                        dtypes=[dtype],
                        example_initializers=[init.name],
                        constraints=["low unique row ratio", "indices or table axis must be shrinkable"],
                        test_fixture="tests/test_mutation_generation.py::test_gather_shrink_remaps_indices",
                        metadata={"shape": list(arr.shape), "unique_row_ratio": unique_ratio, "consumers": consumer_ops},
                    )
                )
    for (_dtype, _shape, _raw), names in signatures.items():
        if len(names) > 1:
            hits.append(
                _hit(
                    "duplicate_initializer_payload",
                    task_id,
                    model_path,
                    len(names) - 1,
                    "initializer_dedup",
                    "safe_build_checked",
                    0.98,
                    0.03,
                    example_initializers=names[:5],
                    constraints=["identical dtype, shape, and bytes"],
                    test_fixture="tests/test_mutation_generation.py::test_initializer_dedup_discovers_duplicates",
                    metadata={"duplicate_count": len(names)},
                )
            )
    return hits


def _zero_initializer_rewrite_family(consumer_ops: list[str]) -> str:
    ops = set(consumer_ops)
    if ops and ops <= BROADCAST_SAFE_OPS:
        return "broadcast_scalarize"
    if "Concat" in ops:
        return "concat_pad"
    if {"ScatterND", "ScatterElements"} & ops:
        return "zero_base_scatter_compiler"
    return "consumer_specific_zero_rewrite"


def _intermediate_hits(task_id: str, model_path: Path, model: onnx.ModelProto) -> list[PatternHit]:
    graph_io = {value.name for value in list(model.graph.input) + list(model.graph.output)}
    shapes = get_value_info_shapes(model)
    producers = get_producers(model)
    consumers = get_consumers(model)
    hits: list[PatternHit] = []
    for name, shape in shapes.items():
        if name in graph_io:
            continue
        numel = _shape_numel(shape)
        elem_type = get_tensor_elem_type(model, name)
        if numel is None or elem_type is None:
            continue
        dtype = _dtype_name(elem_type)
        bytes_est = int(numel * tensor_dtype_size(elem_type))
        producer = producers.get(name)
        consumer_ops = sorted({node.op_type for node in consumers.get(name, [])})
        if elem_type == TensorProto.FLOAT and numel >= 900:
            hits.append(
                _hit(
                    "large_float_intermediate",
                    task_id,
                    model_path,
                    bytes_est,
                    "dtype_lowering|cost_aware_compiler",
                    "requires_operator_probe",
                    0.85,
                    0.45,
                    operators=([producer.op_type] if producer else []) + consumer_ops,
                    dtypes=[dtype],
                    example_nodes=[f"{producer.op_type}:{producer.name or name}" if producer else name],
                    constraints=["graph input/output excluded", "producer and consumers must preserve exact dtype semantics"],
                    test_fixture="new compiler-backed dtype/intermediate fixture",
                    metadata={"tensor": name, "shape": shape, "bytes": bytes_est},
                )
            )
        if elem_type in {TensorProto.BOOL, TensorProto.UINT8} and numel >= 2700:
            hits.append(
                _hit(
                    f"large_{dtype.lower()}_mask_intermediate",
                    task_id,
                    model_path,
                    bytes_est,
                    "bool_mask_rewrite|rule_compiler",
                    "requires_behavior_validation",
                    0.80,
                    0.50,
                    operators=([producer.op_type] if producer else []) + consumer_ops,
                    dtypes=[dtype],
                    example_nodes=[f"{producer.op_type}:{producer.name or name}" if producer else name],
                    constraints=["graph input/output excluded", "prefer symbolic mask over dense one-hot volume"],
                    test_fixture="new symbolic mask compiler fixture",
                    metadata={"tensor": name, "shape": shape, "bytes": bytes_est},
                )
            )
        if producer and producer.op_type == "Equal" and numel >= 300 and len(shape or []) >= 4:
            hits.append(
                _hit(
                    "one_hot_volume_construction",
                    task_id,
                    model_path,
                    bytes_est,
                    "rule_decomposition_compiler",
                    "requires_compiler_backend",
                    0.75,
                    0.60,
                    operators=["Equal"] + consumer_ops,
                    dtypes=[dtype],
                    example_nodes=[_node_label(producer, list(model.graph.node).index(producer))],
                    constraints=["detect channel label comparison", "replace dense one-hot state with scalar/grid state"],
                    test_fixture="new one-hot symbolic compiler fixture",
                    metadata={"tensor": name, "shape": shape, "bytes": bytes_est},
                )
            )
    return hits


def _selector_hits(task_id: str, model_path: Path, model: onnx.ModelProto) -> list[PatternHit]:
    hits: list[PatternHit] = []
    for idx, node in enumerate(model.graph.node):
        if node.op_type == "Where":
            hits.append(
                _hit(
                    "selector_branch_family",
                    task_id,
                    model_path,
                    1,
                    "where_ablation|where_combo|selector_branch_restoration",
                    "operator_supported_behavior_risky",
                    0.70,
                    0.65,
                    operators=["Where"],
                    example_nodes=[_node_label(node, idx)],
                    constraints=["branch choice may be input-dependent", "official validation and symbolic repair required"],
                    test_fixture="tests/test_mutation_generation.py::test_where_combo_applies_multiple_choices",
                    metadata={"inputs": list(node.input), "outputs": list(node.output)},
                )
            )
    return hits


def _arange_hits(task_id: str, model_path: Path, model: onnx.ModelProto) -> list[PatternHit]:
    hits: list[PatternHit] = []
    for init in model.graph.initializer:
        name = init.name.lower()
        if not any(token in name for token in ("arange", "row", "col", "range")):
            continue
        try:
            arr = numpy_helper.to_array(init)
        except Exception:
            continue
        if arr.size < 10 or not np.issubdtype(arr.dtype, np.number):
            continue
        hits.append(
            _hit(
                f"arange_work_dimension_{str(arr.dtype)}",
                task_id,
                model_path,
                int(arr.size),
                "dimension_sweep|dynamic_to_static_compiler",
                "shape_checked_when_applied",
                0.70,
                0.55,
                dtypes=[str(arr.dtype)],
                example_initializers=[init.name],
                constraints=["work dimension must be over-allocated", "official validation required after sweep"],
                test_fixture="tests/test_mutation_generation.py::test_dimension_sweep_discovers_task014_axis_reduction",
                metadata={"shape": list(arr.shape), "min": _json_scalar(arr.min(initial=0)), "max": _json_scalar(arr.max(initial=0))},
            )
        )
    return hits


def model_pattern_hits(
    task_id: str,
    model_path: Path,
    model: onnx.ModelProto,
    ctx: TaskContext,
    include_family_discovery: bool = True,
) -> list[PatternHit]:
    hits: list[PatternHit] = []
    if include_family_discovery:
        hits.extend(_family_hits(task_id, model_path, model, ctx))
    hits.extend(_concat_hits(task_id, model_path, model))
    hits.extend(_initializer_hits(task_id, model_path, model))
    hits.extend(_intermediate_hits(task_id, model_path, model))
    hits.extend(_selector_hits(task_id, model_path, model))
    hits.extend(_arange_hits(task_id, model_path, model))
    return hits


def rank_pattern_hits(hits: list[PatternHit]) -> list[PatternOpportunity]:
    grouped: dict[str, list[PatternHit]] = defaultdict(list)
    for hit in hits:
        grouped[hit.pattern_id].append(hit)
    opportunities: list[PatternOpportunity] = []
    for pattern_id, group in grouped.items():
        tasks = sorted({hit.task_id for hit in group})
        savings_by_task: Counter[str] = Counter()
        for hit in group:
            savings_by_task[hit.task_id] += hit.estimated_savings
        sentinel = max(tasks, key=lambda task: (savings_by_task[task], task)) if tasks else ""
        estimated = int(sum(hit.estimated_savings for hit in group))
        confidence = float(sum(hit.static_shape_confidence for hit in group) / max(1, len(group)))
        risk = float(sum(hit.risk for hit in group) / max(1, len(group)))
        rank_score = len(tasks) * 100000.0 + estimated * max(0.05, confidence) - risk * 25000.0
        families = Counter(hit.rewrite_family for hit in group)
        legality = Counter(hit.ort_legality for hit in group)
        ops = sorted({op for hit in group for op in hit.operators})
        dtypes = sorted({dtype for hit in group for dtype in hit.dtypes})
        nodes = []
        inits = []
        constraints = []
        for hit in group:
            nodes.extend(hit.example_nodes)
            inits.extend(hit.example_initializers)
            constraints.extend(hit.constraints)
        fixture = next((hit.test_fixture for hit in group if hit.test_fixture), "")
        opportunities.append(
            PatternOpportunity(
                pattern_id=pattern_id,
                affected_task_count=len(tasks),
                affected_tasks=tasks,
                estimated_savings=estimated,
                rewrite_family=families.most_common(1)[0][0],
                ort_legality=legality.most_common(1)[0][0],
                static_shape_confidence=round(confidence, 3),
                risk=round(risk, 3),
                rank_score=round(rank_score, 3),
                sentinel_task=sentinel,
                operators=ops[:30],
                dtypes=dtypes[:20],
                example_nodes=list(dict.fromkeys(nodes))[:20],
                example_initializers=list(dict.fromkeys(inits))[:20],
                constraints=list(dict.fromkeys(constraints))[:20],
                test_fixture=fixture,
                example_hits=[asdict(hit) for hit in group[:10]],
            )
        )
    opportunities.sort(key=lambda item: (item.rank_score, item.affected_task_count, item.estimated_savings), reverse=True)
    return opportunities


def mine_patterns(
    repo_root: Path,
    tasks: list[str],
    include_family_discovery: bool = True,
) -> tuple[list[PatternOpportunity], list[dict[str, Any]]]:
    hits: list[PatternHit] = []
    errors: list[dict[str, Any]] = []
    cfg = OptimizerConfig(repo_root=repo_root, max_candidates_per_family=100000)
    for task_id in tasks:
        task_dir = repo_root / task_id
        model_path, source = accepted_incumbent_path(task_dir)
        try:
            model = load_model(model_path)
            baseline_cost = read_baseline_cost(task_dir)
            ctx = TaskContext(
                task_id=task_id,
                task_dir=task_dir,
                baseline_model_path=task_dir / "task" / f"{task_id}.onnx",
                baseline_stats={},
                experiments_dir=task_dir / "experiments",
                run_dir=task_dir / "experiments" / "pattern_miner",
                incumbent_model_path=model_path,
                incumbent_cost=baseline_cost,
                config=cfg,
                baseline_cost=baseline_cost,
            )
            task_hits = model_pattern_hits(task_id, model_path, model, ctx, include_family_discovery=include_family_discovery)
            for hit in task_hits:
                hit.metadata.setdefault("incumbent_source", source)
            hits.extend(task_hits)
        except Exception as exc:
            errors.append({"task": task_id, "model_path": str(model_path), "error": repr(exc)})
    return rank_pattern_hits(hits), errors


def write_markdown(path: Path, opportunities: list[PatternOpportunity], errors: list[dict[str, Any]], max_patterns: int) -> None:
    lines = [
        "# Cross-Task Pattern Miner Report",
        "",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- opportunities: {len(opportunities)}",
        f"- errors: {len(errors)}",
        "",
        "## Top Opportunities",
        "",
    ]
    for idx, item in enumerate(opportunities[:max_patterns], 1):
        lines.extend(
            [
                f"### {idx}. `{item.pattern_id}`",
                "",
                f"- affected tasks: {item.affected_task_count}",
                f"- estimated savings: {item.estimated_savings}",
                f"- rewrite family: `{item.rewrite_family}`",
                f"- ORT legality: {item.ort_legality}",
                f"- static-shape confidence: {item.static_shape_confidence}",
                f"- risk: {item.risk}",
                f"- sentinel task: `{item.sentinel_task}`",
                f"- fixture: {item.test_fixture or 'missing'}",
                f"- example tasks: {', '.join(item.affected_tasks[:12])}",
                f"- operators: {', '.join(item.operators[:12]) or 'n/a'}",
                f"- dtypes: {', '.join(item.dtypes[:12]) or 'n/a'}",
                f"- constraints: {'; '.join(item.constraints[:5]) or 'n/a'}",
                "",
            ]
        )
    if errors:
        lines.extend(["## Errors", ""])
        for err in errors[:50]:
            lines.append(f"- `{err.get('task')}`: {err.get('error')}")
    path.write_text("\n".join(lines) + "\n")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mine cross-task ONNX graph patterns and rank reusable rewrite opportunities")
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--task")
    p.add_argument("--tasks")
    p.add_argument("--all", action="store_true")
    p.add_argument("--top-cost", type=int)
    p.add_argument("--min-cost", type=int)
    p.add_argument("--preset")
    p.add_argument("--run-id")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--max-patterns", type=int, default=50)
    p.add_argument("--min-affected", type=int, default=1)
    p.add_argument("--no-family-discovery", action="store_true")
    p.add_argument("--json", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    selected = discover_tasks(
        repo_root,
        all_tasks=args.all,
        tasks=args.tasks or args.task,
        top_cost=args.top_cost,
        min_cost=args.min_cost,
        preset=args.preset,
    )
    if not selected:
        raise SystemExit("no tasks selected; pass --all, --tasks, --task, or --preset")
    run_id = args.run_id or ("pattern_miner_" + time.strftime("%Y%m%d_%H%M%S"))
    output_dir = args.output_dir or repo_root / ".chatgpt/operations/neurogolf_optimizer/runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    opportunities, errors = mine_patterns(repo_root, selected, include_family_discovery=not args.no_family_discovery)
    opportunities = [item for item in opportunities if item.affected_task_count >= args.min_affected]
    payload = {
        "run_id": run_id,
        "tasks": selected,
        "task_count": len(selected),
        "opportunities": [asdict(item) for item in opportunities],
        "errors": errors,
    }
    write_json(output_dir / "pattern_report.json", payload)
    write_json(output_dir / "top_opportunities.json", [asdict(item) for item in opportunities[: args.max_patterns]])
    write_markdown(output_dir / "PATTERN_REPORT.md", opportunities, errors, args.max_patterns)
    summary = {
        "run_dir": str(output_dir),
        "tasks": len(selected),
        "opportunities": len(opportunities),
        "errors": len(errors),
        "top": [
            {
                "pattern_id": item.pattern_id,
                "affected_tasks": item.affected_task_count,
                "estimated_savings": item.estimated_savings,
                "rewrite_family": item.rewrite_family,
                "sentinel_task": item.sentinel_task,
            }
            for item in opportunities[: min(10, args.max_patterns)]
        ],
    }
    print(json.dumps(payload if args.json else summary, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
