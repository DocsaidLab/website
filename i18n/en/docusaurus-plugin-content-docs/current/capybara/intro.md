---
sidebar_position: 1
---

# Introduction

Capybara primarily consists of the following components:

- **Vision**: Includes computer vision-related functionalities, such as image and video processing.
- **Structures**: Modules for handling structured data, such as BoundingBox and Polygon.
- **ONNXEngine**: Provides ONNX inference functionality, supporting ONNX format models.
- **Utils**: Contains various utility functions for system information, file handling, and other auxiliary tasks.
- **Tests**: Test files used for verifying the functionality of various functions.

## Vision

The Vision module focuses on processing image and video data, offering a rich set of computer vision tools.

Directory structure:

```
vision
├── functionals.py       # Basic image processing functions, such as filtering and transformations
├── geometric.py         # Geometric processing functions, such as rotation and scaling
├── improc.py            # Core image processing logic
├── ipcam                # Module for handling network camera streams
├── morphology.py        # Morphological processing functions, such as dilation and erosion
├── videotools           # Video-related tools
└── visualization        # Visualization tools, such as drawing frames and annotations
```

Main features:

- Image and video reading, processing, and visualization.
- Supports various formats and sources (e.g., local files, network cameras).

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
- Supports various operations like intersection, union, scaling, etc.

## ONNXEngine

The ONNXEngine module provides functionalities related to ONNX model inference.

Directory structure:

```
onnxengine
├── engine.py            # Core inference logic
├── __init__.py          # Initialization file
└── metadata.py          # Model metadata management
```

Main features:

- Supports loading and inference of ONNX models.

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

This is a preliminary introduction to the modules of Capybara.

For specific usage, you can continue reading the corresponding API documentation and sample code.
