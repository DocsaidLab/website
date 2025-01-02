---
sidebar_position: 3
---

# Quick Start

We provide a simple model inference interface, which includes the logic for preprocessing and postprocessing.

First, you need to import the necessary dependencies and create an instance of the `MRZScanner` class.

## Model Inference

:::info
We have designed an automatic model download feature. When the program detects that you are missing a model, it will automatically connect to our server to download it.
:::

Here is a simple example:

```python
import cv2
from skimage import io
from mrzscanner import MRZScanner

# Build model
model = MRZScanner()

# Read image
img = io.imread('https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Inference
result_mrz, error_msg = model(img)

# Output MRZ block with two lines of text and error message
print(result_mrz)
# >>> ('PCAZEQAQARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#     'C946302620AZE6707297F23031072W12IMJ<<<<<<<40')
print(error_msg)
# >>> <ErrorCodes.NO_ERROR: 'No error.'>
```

:::tip
In the above example, the image download link can be found here: [**midv2020_test_mrz.jpg**](https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg)

<div align="center" >
<figure style={{width: "30%"}}>
![test_mrz](./resources/test_mrz.jpg)
</figure>
</div>
:::

## Using the `do_center_crop` Parameter

This image was likely taken with a mobile device and has an elongated shape. If we directly use the model for inference, it may cause excessive text distortion. Therefore, when calling the model, we enable the `do_center_crop` parameter as shown below:

```python
from mrzscanner import MRZScanner

model = MRZScanner()

result, msg = model(img, do_center_crop=True)
print(result)
# >>> ('PCAZEQAOARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#      'C946302620AZE6707297F23031072W12IMJ<<<<<<<40')
print(msg)
# >>> <ErrorCodes.NO_ERROR: 'No error.'>
```

:::tip
`MRZScanner` is wrapped with `__call__`, so you can directly invoke the instance for inference.
:::

## Using with `DocAligner`

Looking at the output above, we notice that although we applied `do_center_crop`, there are still some typos.

This is because we used full image scanning, and the model may have misinterpreted some text in the image.

To improve accuracy, we integrate `DocAligner` to help us align the MRZ block:

```python
import cv2
from docaligner import DocAligner  # Import DocAligner
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
# >>> ('PCAZEQAQARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#      'C946302620AZE6707297F23031072W12IMJ<<<<<<<40')
```

After using `DocAligner`, there's no need to use the `do_center_crop` parameter anymore.

Now, you'll see that the output is more accurate, and the MRZ block from the image has been successfully recognized.

## Error Messages

To help users understand what went wrong, we designed the `ErrorCodes` class.

When the model encounters an error during inference, an error message will be returned, covering the following range:

```python
class ErrorCodes(Enum):
    NO_ERROR = 'No error.'
    INVALID_INPUT_FORMAT = 'Invalid input format.'
    POSTPROCESS_FAILED_LINE_COUNT = 'Postprocess failed, number of lines not 2 or 3.'
    POSTPROCESS_FAILED_TD1_LENGTH = 'Postprocess failed, length of lines not 30 when `doc_type` is TD1.'
    POSTPROCESS_FAILED_TD2_TD3_LENGTH = 'Postprocess failed, length of lines not 36 or 44 when `doc_type` is TD2 or TD3.'
```

This will filter basic errors, such as invalid input format or incorrect line counts.

## Check Digit

Check digits in MRZ are crucial for ensuring data accuracy, used to validate numerical values and prevent data entry errors.

- For a detailed process, we have written about it in the [**Reference: Check Digit**](./reference#檢查碼).

---

In this section, we want to clarify:

- **We do not provide check digit calculation functionality!**

This is because the calculation methods for MRZ check digits are not standardized. Besides the regular check digit calculation methods, MRZ in different regions may define their own check digit calculation methods, so providing a specific check digit calculation method may limit users' flexibility.

:::info
Fun fact:
The check digit for foreign residence permits in Taiwan differs from the global standard. Without collaboration with the government to develop the method, it’s not possible to determine how the check digit is calculated.
:::

Our goal is to train a model focused on MRZ recognition, where each output is automatically determined by the model's format. There are many other open-source projects for check digit calculation, such as the one provided in [**Arg0s1080/mrz**](https://github.com/Arg0s1080/mrz), and we recommend that users directly use this project.
