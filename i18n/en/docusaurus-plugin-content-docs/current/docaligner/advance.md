---
sidebar_position: 4
---

# Advanced

When invoking the `DocAligner` model, you can make advanced settings by passing parameters.

## Initialization

Here are the advanced options available during the initialization phase:

### 1. Backend

Backend is an enumeration type used to specify the computational backend for `DocAligner`.

It includes the following options:

- **cpu**: Use the CPU for computation.
- **cuda**: Use the GPU for computation (requires appropriate hardware support).

```python
from capybara import Backend

model = DocAligner(backend=Backend.cuda) # Using CUDA backend
#
# Or
#
model = DocAligner(backend=Backend.cpu) # Using CPU backend
```

We use ONNXRuntime as the inference engine for the model. Although ONNXRuntime supports various backend engines (including CPU, CUDA, OpenCL, DirectX, TensorRT, etc.), due to common usage environments, we've encapsulated it slightly, currently only offering CPU and CUDA backend engines. Additionally, using the CUDA computation not only requires appropriate hardware but also the installation of corresponding CUDA drivers and CUDA Toolkit.

If CUDA is not installed in your system, or if the installed version is incorrect, the CUDA computation backend cannot be used.

:::tip

1. If you have other requirements, please refer to the [**ONNXRuntime Official Documentation**](https://onnxruntime.ai/docs/execution-providers/index.html) for customization.
2. For issues related to dependency installation, please refer to the [**ONNXRuntime Release Notes**](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)
   :::

### 2. ModelType

ModelType is an enumeration type used to specify the type of model used by `DocAligner`.

It includes the following options:

- **heatmap**: Use the heatmap model.
- **point**: Use the point regression model.

We offer two different models: "heatmap model" and "point regression model."

You can specify the model to use via the `model_type` parameter.

```python
from docaligner import ModelType

model = DocAligner(model_type=ModelType.heatmap) # Using the heatmap model
#
# Or
#
model = DocAligner(model_type=ModelType.point) # Using the point regression model
```

:::tip
However, it's advised not to use the "point regression" model, as its performance is not very satisfactory; this is purely for research purposes.
:::

### 3. ModelCfg

We have trained many models and named them,

You can use `list_models` to see all available models.

```python
from docaligner import DocAligner

print(DocAligner().list_models())
# >>> [
#     'lcnet100',
#     'fastvit_t8',
#     'fastvit_sa24',       <-- Default
#     ...
# ]
```

You can specify the model configuration using the `model_cfg` parameter.

```python
model = DocAligner(model_cfg='fastvit_t8') # Using 'fastvit_t8' configuration
```

## Inference

Here are the advanced settings options during the inference phase:

### CenterCrop

Setting appropriate advanced options during the inference phase can significantly affect model performance and effectiveness.

Among them, `do_center_crop` is a key parameter that determines whether to perform center cropping during inference.

This setting is particularly important because in real-world applications, the images we encounter are often not in standard square sizes.

In reality, image dimensions and proportions vary greatly, such as:

- Photos taken by mobile phones commonly adopt a 9:16 aspect ratio;
- Scanned documents often appear in A4 paper ratios;
- Screenshots are mostly in a 16:9 aspect ratio;
- Images captured through webcams are usually in a 4:3 ratio.

These non-square images, when used directly for inference without proper processing, often contain large areas of irrelevant regions or whitespace, which adversely affect the model's inference effectiveness. Performing center cropping can effectively reduce these irrelevant areas, focusing on the central region of the image, thereby improving the accuracy and efficiency of inference.

Usage is as follows:

```python
from capybara import imread
from docaligner import DocAligner

model = DocAligner()

img = imread('path/to/image.jpg')
result = model(img, do_center_crop=True) # Using center cropping
```

:::tip
**When to use**: Use center cropping when "not cutting the image" and when the image ratio is not square.
:::

:::warning
Center cropping is just one step in the computation process and does not modify the original image. The final results will be mapped back to the original image size, so users need not worry about image distortion or loss of quality.
:::
