Ultralytics YOLOv8 — BiFPN Integration (Custom Fork)
====================================================

This fork customizes Ultralytics YOLOv8 by adding a BiFPN-style neck and utilities, along with a model config and a simple training script to exercise the new components.

What’s Implemented
------------------
- BiFPN module: lightweight, stackable Bi-directional FPN with learnable weights.
  - ultralytics/nn/modules/bifpn.py
- WeightedAdd: normalized learnable fusion for summing multi-scale features.
  - ultralytics/nn/modules/weightedAdd.py
- GetItem: small helper to index list outputs in model graphs.
  - ultralytics/nn/modules/getitem.py
- Depth-wise separable conv block used in the head.
  - ultralytics/nn/modules/dwConv.py
- Model config defining a BiFPN-style head for YOLOv8.
  - ultralytics/cfg/models/v8/yolov8_bifpn.yaml
- Wiring into the model registry so YAML graphs can reference new modules.
  - ultralytics/nn/modules/__init__.py
  - ultralytics/nn/tasks.py
- Minimal training script that loads the new YAML.
  - test_bifpn.py

How To Run
----------
- Python script:
  - `python test_bifpn.py`
    - Trains on `datasets/coco128.yaml` for 100 epochs at `imgsz=640`. Adjust dataset path or device as needed.

- Ultralytics CLI (alternative):
  - `yolo detect train model=ultralytics/cfg/models/v8/yolov8_bifpn.yaml data=datasets/coco128.yaml epochs=100 imgsz=640`

Notes
-----
- BiFPN block initializes learnable fusion weights; explicit initialization can be added for extra stability if desired.
- The custom `DWConv` expects equal in/out channels in the BiFPN head.
- The YAML builds a BiFPN-style neck using `WeightedAdd` and depth-wise convs; you can stack or modify for ablations.
- Ensure Ultralytics dependencies are installed locally and run from this repository root so local modules are used.

Acknowledgement
---------------
Built on top of the Ultralytics YOLOv8 codebase. See the upstream project for full documentation and licensing details.

