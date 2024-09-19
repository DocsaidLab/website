---
sidebar_position: 4
---

# Advanced

When invoking the `MRZScanner` model, you can adjust advanced settings by passing specific parameters.

## Initialization

Here are the advanced settings options available during the initialization phase:

### 1. Backend

`Backend` is an enumerated type used to specify the computation backend for `MRZScanner`.

It includes the following options:

- **cpu**: Uses the CPU for computation.
- **cuda**: Uses the GPU for computation (requires appropriate hardware support).

```python
from docsaidkit import Backend

model = MRZScanner(backend=Backend.cuda)  # Use CUDA backend
#
# or
#
model = MRZScanner(backend=Backend.cpu)  # Use CPU backend
```

We use ONNXRuntime as the inference engine for the model. Although ONNXRuntime supports multiple backends (including CPU, CUDA, OpenCL, DirectX, TensorRT, and more), we’ve streamlined the implementation for regular use environments, currently only offering CPU and CUDA backends. In addition to the necessary hardware support, using the CUDA backend also requires installing the corresponding CUDA drivers and toolkit.

If CUDA is not installed on your system or if the version is incorrect, the CUDA backend will not function.

:::tip

1. For other backend options, refer to the [**ONNXRuntime official documentation**](https://onnxruntime.ai/docs/execution-providers/index.html).
2. For dependency installation details, check the [**ONNXRuntime Release Notes**](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).
   :::

### 2. ModelType

`ModelType` is an enumerated type used to specify the type of model used by `MRZScanner`.

It includes the following option:

- **spotting**: Uses an end-to-end model architecture.

You can specify the model using the `model_type` parameter.

```python
from mrzscanner import MRZScanner

model = MRZScanner(model_type=MRZScanner.spotting)
```

### 3. ModelCfg

You can view all available models using the `list_models` function.

```python
from mrzscanner import MRZScanner

print(MRZScanner().list_models())
# >>> ['20240919']
```

You can specify the model configuration using the `model_cfg` parameter.

```python
model = MRZScanner(model_cfg='20240919')  # Use the '20240919' configuration
```

## Inference

Below are the advanced settings options available during the inference phase:

### Center Cropping

During inference, adjusting certain advanced options can significantly impact the model’s performance and accuracy.

One key parameter is `do_center_crop`, which determines whether to apply center cropping during inference.

This setting is particularly important because real-world images are often not in a standard square format.

In practice, images come in various sizes and aspect ratios, such as:

- Photos taken with a smartphone, typically in a 9:16 aspect ratio.
- Scanned documents often in an A4 paper ratio.
- Webpage screenshots commonly in a 16:9 aspect ratio.
- Images taken with a webcam, usually in a 4:3 ratio.

When these non-square images are directly fed into the model without appropriate preprocessing, irrelevant areas or blank spaces can negatively impact the model’s inference performance. Applying center cropping can effectively reduce these irrelevant areas, allowing the model to focus on the central part of the image, thereby improving both accuracy and efficiency.

Here’s how to use it:

```python
import docsaidkit as D
from mrzscanner import MRZScanner

model = MRZScanner()

img = D.imread('path/to/image.jpg')
result = model(img, do_center_crop=True)  # Apply center cropping
```

:::tip
**When to use**: Use center cropping when the image is not square and will not cut off the MRZ area.
:::

### Post-Processing

In addition to center cropping, we offer a post-processing option to further improve model accuracy. By default, this parameter is set to `do_postprocess=True`.

This is because MRZ blocks follow certain rules, such as country codes being restricted to uppercase letters, and gender being limited to `M` and `F`. These rules can be used to standardize MRZ blocks.

We perform manual corrections for the fields that can be standardized. For instance, in the following code snippet, we replace characters that were mistakenly recognized as digits in fields where only letters should appear:

```python
import re

def replace_digits(text: str):
    text = re.sub('0', 'O', text)
    text = re.sub('1', 'I', text)
    text = re.sub('2', 'Z', text)
    text = re.sub('4', 'A', text)
    text = re.sub('5', 'S', text)
    text = re.sub('8', 'B', text)
    return text

if doc_type == 3:  # TD1
    if len(results[0]) != 30 or len(results[1]) != 30 or len(results[2]) != 30:
        return [''], ErrorCodes.POSTPROCESS_FAILED_TD1_LENGTH
    # Line1
    doc = results[0][0:2]
    country = replace_digits(results[0][2:5])
```

While in our case, this post-processing step did not significantly improve overall accuracy, it can help correct recognition errors in certain scenarios.

If you want to receive raw recognition results, you can set `do_postprocess` to `False` during inference:

```python
result, msg = model(img, do_postprocess=False)
```
