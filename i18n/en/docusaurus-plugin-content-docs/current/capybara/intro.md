---
sidebar_position: 1
---

# Introduction

Capybara consists of the following parts (based on the current code structure):

- **Vision** (`capybara.vision`): Image/video I/O and processing.
- **Structures** (`capybara.structures`): Geometric structures such as `Box/Boxes`, `Polygon/Polygons`, and `Keypoints`.
- **Runtime** (`capybara.runtime`): Registry and selection logic for runtime/backend.
- **Inference engines (optional)**:
  - `capybara.onnxengine` (ONNXRuntime)
  - `capybara.openvinoengine` (OpenVINO)
  - `capybara.torchengine` (TorchScript)
- **Utils** (`capybara.utils`): Utilities (paths, downloads, time, etc.).
- **Extras (optional)**:
  - `visualization`: drawing utilities (`capybara.vision.visualization`)
  - `ipcam`: simple Web demo (`capybara.vision.ipcam`)
  - `system`: system info utilities (`capybara.utils.system_info`)

## Vision

The Vision module focuses on image/video processing and I/O.

Directory structure:

```
vision
├── functionals.py       # Basic image processing functions, such as filtering and transformations
├── geometric.py         # Geometric processing functions, such as rotation and scaling
├── improc.py            # Core image processing logic
├── morphology.py        # Morphological processing functions, such as dilation and erosion
├── videotools           # Video-related tools
├── ipcam                # (extra: ipcam) IPCam demo
└── visualization        # (extra: visualization) drawing/visualization
```

Main features:

- Image/video reading, processing, and visualization.
- Supports multiple sources (local files, video frame extraction, IPCam demo, etc.).

## Structures

The Structures module is responsible for handling structured data, commonly used in computer vision and data analysis scenarios.

Directory structure:

```
structures
├── functionals.py       # Related functional functions
├── boxes.py             # Box and Boxes data structures
├── keypoints.py         # Keypoints data structure
└── polygons.py          # Polygon and Polygons data structures
```

Main features:

- Provides structured data processing for Boxes, Keypoints, and Polygons.
- Supports operations such as intersection, IoU, scaling, etc.

## Runtime / Inference engines (optional)

Inference-related features live in separate modules, and `capybara.runtime` provides a unified way to describe/choose runtime/backend.

Note: inference backends are optional dependencies. Install the required extras first (e.g. `capybara-docsaid[onnxruntime]`).

### capybara.runtime

- Defines `Runtime` / `Backend`, and provides selection helpers such as `auto_backend_name()`.

### capybara.onnxengine

Directory structure:

```
onnxengine
├── engine.py            # Core inference logic
├── __init__.py          # Initialization file
├── metadata.py          # Model metadata management
└── utils.py             # ONNX helpers
```

Main features:

- Load and run inference on ONNX models.

### capybara.openvinoengine

- OpenVINO inference wrapper (sync inference + optional async queue).

### capybara.torchengine

- TorchScript inference wrapper (simple dtype/device normalization).

## Utils

The Utils module contains a wide range of auxiliary utility functions.

Directory structure:

```
utils
├── custom_path.py       # Custom path operations
├── custom_tqdm.py       # Progress bar utility
├── files_utils.py       # File handling functions
├── powerdict.py         # Enhanced dictionary operations
├── system_info.py       # System information detection
├── time.py              # Time handling utilities
└── utils.py             # General utility functions
```

Main features:

- File handling and system information detection.
- Provides custom tools like progress bars and enhanced dictionaries.

## Tests

The Tests module is used to verify that the system functions correctly.

Main features:

- Includes unit tests for various modules.
- Provides fast regression and functionality verification.

---

This is a high-level introduction to the modules of Capybara.

For API usage and examples, continue reading the docs. If you see an import error, check whether the corresponding extra is installed.
