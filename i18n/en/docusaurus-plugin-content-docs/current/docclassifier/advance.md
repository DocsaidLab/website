---
sidebar_position: 4
---

# Advanced

When invoking the `DocClassifier` model, you can perform advanced settings by passing parameters.

## Initialization

Here are the advanced setting options during initialization:

### 1. Backend

Backend is an enumeration type used to specify the computation backend of `DocClassifier`.

It includes the following options:
- **cpu**: Perform computation using CPU.
- **cuda**: Perform computation using GPU (requires appropriate hardware support).

```python
from docsaidkit import Backend

model = DocClassifier(backend=Backend.cuda) # Use CUDA backend
#
# or
#
model = DocClassifier(backend=Backend.cpu) # Use CPU backend
```

We use ONNXRuntime as the inference engine for the model. Although ONNXRuntime supports multiple backend engines (including CPU, CUDA, OpenCL, DirectX, TensorRT, etc.), due to typical usage environments, we have made a slight encapsulation. Currently, only CPU and CUDA backend engines are provided. In addition, using the CUDA backend for computation requires both appropriate hardware support and the installation of corresponding CUDA drivers and CUDA Toolkit.

If CUDA is not installed on your system or if the version is incorrect, the CUDA backend cannot be used.

:::tip
1. If you have other requirements, please refer to the [**ONNXRuntime official documentation**](https://onnxruntime.ai/docs/execution-providers/index.html) for customization.
2. For issues related to installing dependencies, please refer to the [**ONNXRuntime Release Notes**](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).
:::

### 2. ModelType

ModelType is an enumeration type used to specify the type of model used by `DocClassifier`.

It includes the following options:

- **margin_based**: Use a model architecture based on the margin method.

You can specify the model to use through the `model_type` parameter.

```python
from docclassifier import ModelType

model = DocClassifier(model_type=ModelType.margin_based)
```

### 3. ModelCfg

You can use `list_models` to view all available models.

```python
from docclassifier import DocClassifier

print(DocClassifier().list_models())
# >>> ['20240326']
```

You can specify the model configuration using the `model_cfg` parameter.

```python
model = DocClassifier(model_cfg='20240326') # Use '20240326' configuration
```

## Inference

There are no advanced setting options during the inference phase for this module. More features may be added in future versions.

## Feature Extraction

You may be more interested in the features of the document rather than its classification. For this purpose, we provide the `extract_feature` method.

```python
from docclassifier import DocClassifier
import docsaidkit as D

model = DocClassifier()
img = D.imread('path/to/image.jpg')

# Extract features: Returns a 256-dimensional feature vector
features = model.extract_feature(img)
```