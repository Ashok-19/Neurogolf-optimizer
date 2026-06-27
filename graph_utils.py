from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

from .models import BuildResult


def load_model(path: str | Path) -> onnx.ModelProto:
    return onnx.load(str(path))


def save_model(model: onnx.ModelProto, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def clone_model(model: onnx.ModelProto) -> onnx.ModelProto:
    return copy.deepcopy(model)


def model_hash(path_or_model: str | Path | onnx.ModelProto) -> str:
    if isinstance(path_or_model, onnx.ModelProto):
        data = path_or_model.SerializeToString()
    else:
        data = Path(path_or_model).read_bytes()
    return hashlib.sha256(data).hexdigest()


def infer_shapes(model: onnx.ModelProto, strict: bool = True) -> onnx.ModelProto:
    return shape_inference.infer_shapes(model, strict_mode=strict)


def check_model(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model)


def safe_build(model: onnx.ModelProto, allow_nonstrict_shapes: bool = False) -> BuildResult:
    try:
        sorted_model = topological_sort(model)
        try:
            built = infer_shapes(sorted_model, strict=True)
        except Exception:
            if not allow_nonstrict_shapes:
                raise
            built = infer_shapes(sorted_model, strict=False)
        built = remove_unused_value_info(built)
        check_model(built)
        return BuildResult(model=built, ok=True, changed=True)
    except Exception as exc:
        return BuildResult(model=None, ok=False, error=f"{type(exc).__name__}: {exc}")


def list_nodes(model: onnx.ModelProto) -> list[onnx.NodeProto]:
    return list(model.graph.node)


def list_initializers(model: onnx.ModelProto) -> list[onnx.TensorProto]:
    return list(model.graph.initializer)


def initializer_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for init in model.graph.initializer:
        try:
            out[init.name] = numpy_helper.to_array(init)
        except Exception:
            continue
    return out


def replace_initializer(model: onnx.ModelProto, name: str, array: np.ndarray) -> bool:
    for i, init in enumerate(model.graph.initializer):
        if init.name == name:
            model.graph.initializer.remove(init)
            model.graph.initializer.insert(i, numpy_helper.from_array(array, name=name))
            return True
    return False


def replace_all_inputs(model: onnx.ModelProto, old: str, new: str) -> int:
    count = 0
    for node in model.graph.node:
        for i, value in enumerate(node.input):
            if value == old:
                node.input[i] = new
                count += 1
    for output in model.graph.output:
        if output.name == old:
            output.name = new
            count += 1
    return count


def replace_graph_output_with_identity(model: onnx.ModelProto, src_name: str, output_name: str) -> None:
    if src_name == output_name:
        return
    for node in model.graph.node:
        for i, out in enumerate(node.output):
            if out == output_name:
                node.output[i] = f"{output_name}__optimizer_dead"
    node = helper.make_node("Identity", [src_name], [output_name], name=f"optimizer_identity_{output_name}")
    model.graph.node.append(node)


def find_initializer(model: onnx.ModelProto, name: str) -> onnx.TensorProto | None:
    for init in model.graph.initializer:
        if init.name == name:
            return init
    return None


def remove_initializer(model: onnx.ModelProto, name: str) -> bool:
    for init in list(model.graph.initializer):
        if init.name == name:
            model.graph.initializer.remove(init)
            return True
    return False


def set_initializer(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    for i, init in enumerate(list(model.graph.initializer)):
        if init.name == name:
            model.graph.initializer.remove(init)
            model.graph.initializer.insert(i, numpy_helper.from_array(array, name=name))
            return
    model.graph.initializer.append(numpy_helper.from_array(array, name=name))


def get_consumers(model: onnx.ModelProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)
    return consumers


def get_producers(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    producers: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for out in node.output:
            if out:
                producers[out] = node
    return producers


def get_node_by_output(model: onnx.ModelProto, output_name: str) -> onnx.NodeProto | None:
    return get_producers(model).get(output_name)


def _dims_from_vi(value_info: Any) -> list[int | None]:
    dims = []
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return dims
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        else:
            dims.append(None)
    return dims


def get_value_info_shapes(model: onnx.ModelProto) -> dict[str, list[int | None]]:
    out = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if vi.type.HasField("tensor_type"):
            out[vi.name] = _dims_from_vi(vi)
    for init in model.graph.initializer:
        out.setdefault(init.name, list(init.dims))
    return out


def get_tensor_elem_type(model: onnx.ModelProto, name: str) -> int | None:
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if vi.name == name and vi.type.HasField("tensor_type"):
            return int(vi.type.tensor_type.elem_type)
    for init in model.graph.initializer:
        if init.name == name:
            return int(init.data_type)
    return None


def tensor_numel_from_value_info(value_info: Any) -> int | None:
    dims = _dims_from_vi(value_info)
    if not dims or any(d is None for d in dims):
        return None
    n = 1
    for d in dims:
        n *= int(d)
    return n


def estimate_initializer_params(model: onnx.ModelProto) -> int:
    total = 0
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        total += int(arr.size)
    return total


def remove_unused_initializers(model: onnx.ModelProto) -> onnx.ModelProto:
    used = {inp for node in model.graph.node for inp in node.input if inp}
    used.update(inp.name for inp in model.graph.input)
    keep = [init for init in model.graph.initializer if init.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    return model


def remove_unused_value_info(model: onnx.ModelProto) -> onnx.ModelProto:
    names = {vi.name for vi in list(model.graph.input) + list(model.graph.output)}
    names.update(init.name for init in model.graph.initializer)
    for node in model.graph.node:
        names.update(inp for inp in node.input if inp)
        names.update(out for out in node.output if out)
    keep = [vi for vi in model.graph.value_info if vi.name in names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(keep)
    return model


def prune_unreachable(model: onnx.ModelProto) -> onnx.ModelProto:
    producers = get_producers(model)
    needed_values = {out.name for out in model.graph.output}
    needed_nodes: set[int] = set()
    changed = True
    while changed:
        changed = False
        for idx in range(len(model.graph.node) - 1, -1, -1):
            node = model.graph.node[idx]
            if idx in needed_nodes:
                continue
            if any(out in needed_values for out in node.output):
                needed_nodes.add(idx)
                needed_values.update(inp for inp in node.input if inp)
                changed = True
        for value in list(needed_values):
            producer = producers.get(value)
            if producer is not None:
                needed_values.update(inp for inp in producer.input if inp)
    keep_nodes = [node for idx, node in enumerate(model.graph.node) if idx in needed_nodes]
    del model.graph.node[:]
    model.graph.node.extend(keep_nodes)
    remove_unused_initializers(model)
    remove_unused_value_info(model)
    return model


def topological_sort(model: onnx.ModelProto) -> onnx.ModelProto:
    available = {vi.name for vi in model.graph.input}
    available.update(init.name for init in model.graph.initializer)
    available.update("")
    nodes = list(model.graph.node)
    sorted_nodes: list[onnx.NodeProto] = []
    remaining = nodes[:]
    while remaining:
        progressed = False
        next_remaining = []
        for node in remaining:
            if all(inp in available for inp in node.input if inp):
                sorted_nodes.append(node)
                available.update(out for out in node.output if out)
                progressed = True
            else:
                next_remaining.append(node)
        if not progressed:
            missing = sorted({inp for n in next_remaining for inp in n.input if inp and inp not in available})[:10]
            raise RuntimeError(f"topological sort failed; missing inputs: {missing}")
        remaining = next_remaining
    del model.graph.node[:]
    model.graph.node.extend(sorted_nodes)
    return model


def tensor_dtype_size(elem_type: int) -> int:
    sizes = {
        TensorProto.FLOAT: 4,
        TensorProto.FLOAT16: 2,
        TensorProto.DOUBLE: 8,
        TensorProto.UINT8: 1,
        TensorProto.INT8: 1,
        TensorProto.UINT16: 2,
        TensorProto.INT16: 2,
        TensorProto.INT32: 4,
        TensorProto.INT64: 8,
        TensorProto.BOOL: 1,
    }
    return sizes.get(elem_type, 4)


def initializer_signature(init: onnx.TensorProto) -> tuple[int, tuple[int, ...], str]:
    arr = numpy_helper.to_array(init)
    return int(init.data_type), tuple(init.dims), hashlib.sha256(arr.tobytes()).hexdigest()
