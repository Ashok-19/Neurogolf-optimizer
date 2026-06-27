from __future__ import annotations

from ..dsl import Program, Step


def generate(pairs, features, config):
    out_shapes = {pf.output_shape for pf in features.pairs}
    if len(out_shapes) == 1:
        oh, ow = next(iter(out_shapes))
        yield Program("crop_nonzero_bbox_fixed", "crop", {"output_h": oh, "output_w": ow}, [Step("Input"), Step("CropNonZeroBBox", {"output_h": oh, "output_w": ow})])
    yield Program("crop_nonzero_bbox", "crop", steps=[Step("Input"), Step("CropNonZeroBBox", {"output_h": 30, "output_w": 30})])
    for color in range(1, 10):
        if len(out_shapes) == 1:
            oh, ow = next(iter(out_shapes))
            yield Program(f"crop_color_{color}_bbox_fixed", "crop", {"color": color, "output_h": oh, "output_w": ow}, [Step("Input"), Step("CropColorBBox", {"color": color, "output_h": oh, "output_w": ow})])
        yield Program(f"crop_color_{color}_bbox", "crop", {"color": color}, [Step("Input"), Step("CropColorBBox", {"color": color, "output_h": 30, "output_w": 30})])
    bboxes = [pf.crop_bbox for pf in features.pairs if pf.crop_bbox is not None]
    if bboxes and len(set(bboxes)) == 1:
        yield Program("static_crop_bbox", "crop", {"bbox": bboxes[0]}, [Step("Input"), Step("Crop", {"bbox": bboxes[0]})])
