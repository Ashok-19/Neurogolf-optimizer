# task390 compact-feature QLinearConv rewrite

This is a fresh structural rewrite of the task rule rather than a mutation of the previous ONNX graph.

The solver extracts the red gripper geometry and gray payload into compact 15x15 feature planes, reflects the payload across the nearest red gripper line, and uses a padded final-output `QLinearConv` to distinguish background, red, gray, and implicit padding without constructing a charged 30x30 label tensor.

Validation:

- public wrong: 0
- bundled ARC-GEN wrong: 0/262
- fresh validation: 1000/1000
- baseline cost: 2699
- promoted cost: 2608
- score gain: 0.03429768997146102

The deterministic builder used for the promoted artifact is `build_compact_feature_qconv.py` in the corresponding bucket work directory.
