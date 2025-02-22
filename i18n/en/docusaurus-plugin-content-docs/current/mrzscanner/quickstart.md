---
sidebar_position: 3
---

# Quick Start

We provide a simple model inference interface that includes both preprocessing and postprocessing logic.

## Model Inference

First, don’t worry about anything—just run the following code and check if it executes successfully:

```python
import cv2
from skimage import io
from mrzscanner import MRZScanner

# Create the model
model = MRZScanner()

# Read image from URL
img = io.imread('https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Model inference
result = model(img, do_center_crop=True, do_postprocess=False)

# Output result
print(result)
# {
#     'mrz_polygon':
#         array(
#             [
#                 [ 158.536 , 1916.3734],
#                 [1682.7792, 1976.1683],
#                 [1677.1018, 2120.8926],
#                 [ 152.8586, 2061.0977]
#             ],
#             dtype=float32
#         ),
#     'mrz_texts': [
#         'PCAZEQAQARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#         'C946302620AZE6707297F23031072W12IMJ<<<<<<<40'
#     ],
#     'msg': <ErrorCodes.NO_ERROR: 'No error.'>
# }
```

If it runs successfully, let’s dive into the details of the code below.

:::info
We have designed an automatic model download feature. When the program detects that the model is missing, it will automatically connect to our server and download it.
:::

:::tip
In the example above, the image download link is referenced here: [**midv2020_test_mrz.jpg**](https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg)

<div align="center" >
<figure style={{width: "30%"}}>
![test_mrz](./resources/test_mrz.jpg)
</figure>
</div>
:::

## Using the `do_center_crop` Parameter

This image was likely taken with a mobile device, which is long and narrow. If we directly infer it with the model, it could result in excessive distortion of the text. Therefore, we use the `do_center_crop` parameter during inference. This parameter is used to center-crop the image.

By default, this parameter is set to `False` because we believe that images should not be modified without the user’s knowledge. However, in practical applications, the images we encounter are often not in standard square dimensions.

In reality, images come in many different sizes and aspect ratios, such as:

- Photos taken with mobile phones commonly use a 9:16 aspect ratio;
- Scanned documents typically follow A4 paper size ratios;
- Webpage screenshots often have a 16:9 aspect ratio;
- Images taken by webcams are usually in a 4:3 ratio.

These non-square images, when inferred without proper processing, often contain irrelevant areas or empty spaces, which can negatively affect the model’s inference results. Center cropping helps effectively reduce these irrelevant areas and focus on the central part of the image, thus improving both inference accuracy and efficiency.

Usage:

```python
from mrzscanner import MRZScanner

model = MRZScanner()

result = model(img, do_center_crop=True) # Use center cropping
```

:::tip
**When to use**: Use center cropping when the image ratio is not square and does not crop into the MRZ area.
:::

:::info
`MRZScanner` has already been encapsulated using `__call__`, so you can directly call the instance for inference.
:::

## Using the `do_postprocess` Parameter

In addition to center cropping, we also provide a postprocessing option `do_postprocess` to further improve the accuracy of the model.

This parameter is also set to `False` by default, for the same reason as before: we believe that the recognition result should not be modified without the user's awareness.

In real-world applications, there are rules for the MRZ block, such as: country codes must be uppercase letters, gender can only be `M` or `F`, and fields related to dates must contain only digits. These rules can help regulate the MRZ block.

Therefore, we perform manual corrections on fields that can be standardized. Below is a code snippet that demonstrates the concept of postprocessing, where we replace possibly misrecognized digits with the correct characters in fields where numbers are not expected:

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

Although this postprocessing step did not improve our accuracy in this project, keeping this functionality can still correct misrecognized results in certain situations.

You may want to set `do_postprocess` to `True` during inference for potentially better results:

```python
result = model(img, do_postprocess=True)
```

Alternatively, if you prefer to see the raw output from the model, you can stick with the default value.

## Using with `DocAligner`

Sometimes, even with the `do_center_crop` parameter, detection may still fail. In such cases, we can use `DocAligner` to help find the document's location before performing MRZ recognition.

```python
import cv2
from docaligner import DocAligner # Import DocAligner
from mrzscanner import MRZScanner
from capybara import imwarp_quadrangle
from skimage import io

model = MRZScanner()

doc_aligner = DocAligner()

img = io.imread('https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

polygon = doc_aligner(img)
flat_img = imwarp_quadrangle(img, polygon, dst_size=(800, 480))

print(model(flat_img))
# {
#     'mrz_polygon':
#         array(
#         [
#             [ 34.0408 , 378.497  ],
#             [756.4258 , 385.0492 ],
#             [755.8944 , 443.63843],
#             [ 33.5094 , 437.08618]
#         ], dtype=float32
#     ),
#     'mrz_texts': [
#         'PCAZEQAQARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#         'C946302620AZE6707297F23031072W12IMJ<<<<<<<40'
#     ],
#     'msg': <ErrorCodes.NO_ERROR: 'No error.'>
# }
```

:::warning
When using `DocAligner` for preprocessing, the MRZ may already occupy a certain area of the image. Therefore, you don't need to use the `do_center_crop` parameter, as center cropping may cut off parts of the MRZ.
:::

:::tip
For more details on how to use `DocAligner`, refer to the [**DocAligner Technical Documentation**](https://docsaid.org/en/docs/docaligner/).
:::

## Error Messages

To help users understand the causes of errors, we have designed the `ErrorCodes` class.

When the model encounters an error during inference, it will return an error message, covering issues such as:

```python
class ErrorCodes(Enum):
    NO_ERROR = 'No error.'
    INVALID_INPUT_FORMAT = 'Invalid input format.'
    POSTPROCESS_FAILED_LINE_COUNT = 'Postprocess failed, number of lines not 2 or 3.'
    POSTPROCESS_FAILED_TD1_LENGTH = 'Postprocess failed, length of lines not 30 when `doc_type` is TD1.'
    POSTPROCESS_FAILED_TD2_TD3_LENGTH = 'Postprocess failed, length of lines not 36 or 44 when `doc_type` is TD2 or TD3.'
```

Basic errors, such as incorrect input format or incorrect line count, are filtered here.

## Check Digits

Check digits in the MRZ are crucial for ensuring the accuracy of the data. They are used to verify the correctness of numeric data, preventing data entry errors.

- For detailed procedures, please refer to [**Reference: Check Digits**](./reference#check-digit).

---

In this section, we want to highlight:

- **We do not provide a check digit calculation function!**

This is because the check digit calculation method for MRZ is not uniform. Besides the standard check digit calculation method, MRZs from different regions can define their own calculation methods. Therefore, providing a single check digit calculation method may limit user flexibility.

:::info
Fun fact:

The check digit calculation for Taiwan's foreign resident permits' MRZ differs from the global standard. Without collaboration with the government, the calculation method for this check digit is not publicly available.
:::

Our goal is to train a model focused on MRZ recognition, where each output's format is automatically determined by the model. There are many other open-source projects that provide check digit calculation functionality, such as [**Arg0s1080/mrz**](https://github.com/Arg0s1080/mrz). We recommend users utilize this project for check digit calculations.
