from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from onnx import TensorProto, numpy_helper

from .graph_utils import get_consumers, get_node_by_output, get_value_info_shapes, initializer_arrays, load_model, tensor_dtype_size
from .models import TaskAnalysis


def analyze_model(task_id: str, model_path: Path) -> TaskAnalysis:
    model = load_model(model_path)
    ops = Counter(node.op_type for node in model.graph.node)
    arrays = initializer_arrays(model)
    dtype_hist = Counter()
    largest = []
    lookup = []
    zeros = []
    sigs: dict[tuple, list[str]] = {}
    param_estimate = 0
    for init in model.graph.initializer:
        try:
            arr = numpy_helper.to_array(init)
        except Exception:
            continue
        dtype_name = TensorProto.DataType.Name(init.data_type)
        dtype_hist[dtype_name] += int(arr.size)
        param_estimate += int(arr.size)
        bytes_est = int(arr.size * tensor_dtype_size(init.data_type))
        item = {"name": init.name, "shape": list(init.dims), "dtype": dtype_name, "size": int(arr.size), "bytes": bytes_est}
        largest.append(item)
        sigs.setdefault((int(init.data_type), tuple(init.dims), arr.tobytes()), []).append(init.name)
        if arr.ndim >= 2 and arr.size >= 512:
            unique_ratio = float(len(np.unique(arr.reshape(-1, arr.shape[-1]), axis=0))) / max(1, arr.reshape(-1, arr.shape[-1]).shape[0])
            if unique_ratio < 0.75:
                lookup.append({**item, "unique_row_ratio": unique_ratio})
        if arr.size >= 128 and (np.all(arr == 0) or (arr.dtype == np.bool_ and not arr.any())):
            zeros.append(item)
    shapes = get_value_info_shapes(model)
    float_bytes = 0
    small_tensors = 0
    intermediates = []
    for name, shape in shapes.items():
        elem_type = None
        for vi in list(model.graph.value_info) + list(model.graph.output) + list(model.graph.input):
            if vi.name == name and vi.type.HasField("tensor_type"):
                elem_type = int(vi.type.tensor_type.elem_type)
                break
        if not shape or any(d is None for d in shape) or elem_type is None:
            continue
        n = 1
        for d in shape:
            n *= int(d)
        if elem_type == TensorProto.FLOAT:
            float_bytes += n * 4
        if elem_type in (TensorProto.BOOL, TensorProto.UINT8):
            small_tensors += 1
        if name not in {i.name for i in model.graph.input} and name not in {o.name for o in model.graph.output}:
            intermediates.append({"name": name, "shape": shape, "elem_type": TensorProto.DataType.Name(elem_type), "bytes": int(n * tensor_dtype_size(elem_type))})
    final_pattern = None
    if model.graph.output:
        producer = get_node_by_output(model, model.graph.output[0].name)
        final_pattern = producer.op_type if producer else "input_or_initializer"
    graph_outputs = {o.name for o in model.graph.output}
    where_graph_output_count = sum(1 for n in model.graph.node if n.op_type == "Where" and any(o in graph_outputs for o in n.output))
    consumers = get_consumers(model)
    concat_patterns = []
    for idx, node in enumerate(model.graph.node):
        if node.op_type == "Concat":
            concat_patterns.append({"node_index": idx, "inputs": len(node.input), "output": node.output[0] if node.output else None})
    gather_patterns = []
    for idx, node in enumerate(model.graph.node):
        if node.op_type in {"Gather", "GatherND", "GatherElements"} and len(node.input) >= 2:
            gather_patterns.append({"node_index": idx, "op": node.op_type, "data": node.input[0], "indices": node.input[1]})
    conv_details = []
    for idx, node in enumerate(model.graph.node):
        if node.op_type in {"Conv", "QLinearConv", "ConvTranspose"}:
            conv_details.append({"node_index": idx, "op": node.op_type, "inputs": list(node.input), "outputs": list(node.output), "consumer_count": sum(len(consumers.get(o, [])) for o in node.output)})
    tags = []
    if ops["Where"] >= 3:
        tags.append("where_heavy")
    if lookup:
        tags.append("table_heavy")
    if ops["Conv"] or ops["ConvTranspose"]:
        tags.append("conv_heavy")
    if ops["QLinearConv"]:
        tags.append("qlinear_heavy")
    if ops["Concat"] >= 2 or final_pattern == "Concat":
        tags.append("concat_padding")
    if float_bytes > 10000:
        tags.append("dtype_heavy")
    if ops["Equal"] and ops["Concat"]:
        tags.append("scalar_label")
    if ops["And"] + ops["Or"] + ops["Not"] + ops["Xor"] + ops["Greater"] + ops["Less"] >= 3:
        tags.append("bool_mask")
    if final_pattern in {"Where", "Concat", "Equal"}:
        tags.append("final_selector")
    if len(model.graph.node) <= 2:
        tags.append("one_node")
    if len(model.graph.node) > 50 or sum(x["bytes"] for x in largest) > 10000:
        tags.append("high_cost")
    largest.sort(key=lambda x: x["bytes"], reverse=True)
    intermediates.sort(key=lambda x: x["bytes"], reverse=True)
    duplicate_initializers = [
        {"names": names, "count": len(names)}
        for (_dtype, _shape, _raw), names in sigs.items()
        if len(names) > 1
    ][:50]
    return TaskAnalysis(
        task_id=task_id,
        node_count=len(model.graph.node),
        op_histogram=dict(ops),
        initializer_count=len(model.graph.initializer),
        largest_initializers=largest[:20],
        dtype_histogram=dict(dtype_hist),
        where_count=ops["Where"],
        concat_count=ops["Concat"],
        conv_count=ops["Conv"],
        qlinear_conv_count=ops["QLinearConv"],
        conv_transpose_count=ops["ConvTranspose"],
        gather_count=ops["Gather"] + ops["GatherND"] + ops["GatherElements"],
        scatter_count=ops["ScatterND"] + ops["ScatterElements"],
        estimated_float32_intermediate_bytes=int(float_bytes),
        estimated_bool_uint8_tensors=small_tensors,
        large_lookup_table_candidates=lookup[:20],
        large_zero_false_tensor_candidates=zeros[:20],
        final_output_producer_pattern=final_pattern,
        tags=tags,
        param_estimate=param_estimate,
        duplicate_initializers=duplicate_initializers,
        where_graph_output_count=where_graph_output_count,
        concat_padding_patterns=concat_patterns[:50],
        gather_table_patterns=gather_patterns[:50],
        conv_details=conv_details[:50],
        largest_intermediates=intermediates[:20],
    )


def analyze_task(task_id: str, repo_root: Path | None = None) -> TaskAnalysis:
    root = repo_root or Path.cwd()
    return analyze_model(task_id, root / task_id / "task" / f"{task_id}.onnx")
