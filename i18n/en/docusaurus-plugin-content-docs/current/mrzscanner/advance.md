---
sidebar_position: 4
---

# Advanced

When invoking the `MRZScanner` model, you can make advanced configurations by passing parameters.

## Initialization

The following are advanced configuration options during the initialization phase:

### 1. Backend

The Backend is an enumeration type used to specify the computation backend for `MRZScanner`.

It includes the following options:

- **cpu**: Use CPU for computation.
- **cuda**: Use GPU for computation (requires appropriate hardware support).

```python
from capybara import Backend

model = MRZScanner(backend=Backend.cuda) # Use CUDA backend
#
# Or
#
model = MRZScanner(backend=Backend.cpu) # Use CPU backend
```

We use ONNXRuntime as the inference engine for the model. Although ONNXRuntime supports various backends (including CPU, CUDA, OpenCL, DirectX, TensorRT, etc.), due to the environments typically in use, we have encapsulated it slightly and currently only provide CPU and CUDA backends. Additionally, to use CUDA computation, you need appropriate hardware support and must install the corresponding CUDA drivers and toolkit.

If CUDA is not installed on your system or the version is incorrect, the CUDA computation backend will not be available.

:::tip

1. If you have other requirements, refer to the [**ONNXRuntime official documentation**](https://onnxruntime.ai/docs/execution-providers/index.html) for custom configurations.
2. For installation-related issues, refer to the [**ONNXRuntime Release Notes**](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)
   :::

### 2. ModelType

ModelType is an enumeration type used to specify the type of model used by `MRZScanner`.

It includes the following option:

- **spotting**: Use an end-to-end model architecture.

You can specify the model to use via the `model_type` parameter.

```python
from mrzscanner import MRZScanner

model = MRZScanner(model_type=MRZScanner.spotting)
```

### 3. ModelCfg

You can use `list_models` to see all available models.

```python
from mrzscanner import MRZScanner

print(MRZScanner().list_models())
# >>> ['20240919']
```

You can specify the model configuration using the `model_cfg` parameter.

```python
model = MRZScanner(model_cfg='20240919') # Use '20240919' configuration
```

## Inference

The following are advanced configuration options during the inference phase:

### Center Cropping

During the inference phase, setting appropriate advanced options can significantly affect the model's performance and results.

One critical parameter is `do_center_crop`, which determines whether to perform center cropping during inference.

This setting is particularly important because images encountered in real-world applications are often not in standard square dimensions.

In fact, image sizes and ratios vary, for example:

- Photos taken with a smartphone commonly use a 9:16 aspect ratio;
- Scanned documents usually follow the A4 paper ratio;
- Webpage screenshots are typically 16:9;
- Images taken via webcam are often 4:3.

These non-square images, when directly used for inference without appropriate processing, often contain irrelevant areas or blank spaces, which negatively affect the model's inference performance. Center cropping helps effectively reduce these irrelevant areas by focusing on the central region of the image, thus improving inference accuracy and efficiency.

Usage:

```python
import capybara as cb
from mrzscanner import MRZScanner

model = MRZScanner()

img = cb.imread('path/to/image.jpg')
result = model(img, do_center_crop=True) # Use center cropping
```

:::tip
**When to Use**: Use center cropping when the image aspect ratio is not square and does not crop the MRZ region.
:::

### Post-Processing

In addition to center cropping, we provide a post-processing option to further improve model accuracy. We offer a post-processing parameter, which is set to `do_postprocess=True` by default.

This is because the MRZ block has certain rules, such as country codes being uppercase letters, and gender being either `M` or `F`, etc. These rules can be used to standardize the MRZ block.

Thus, we perform manual corrections for blocks that can be standardized. For example, the following code snippet replaces possible misinterpreted digits with the correct characters in fields where numbers should not appear:

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

While this post-processing step hasn't significantly improved accuracy in our cases, retaining this feature can still help correct erroneous recognition results in certain situations.

You can set `do_postprocess` to `False` during inference to get the raw recognition results.

```python
result, msg = model(img, do_postprocess=False)
```
