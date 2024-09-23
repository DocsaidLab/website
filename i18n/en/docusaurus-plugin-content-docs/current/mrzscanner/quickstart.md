---
sidebar_position: 3
---

# QuickStart

We provide a simple model inference interface that includes preprocessing and postprocessing logic.

First, you need to import the necessary dependencies and create an instance of the `MRZScanner` class.

## Model Inference

Below is a simple example demonstrating how to use `MRZScanner` for model inference:

```python
from mrzscanner import MRZScanner

model = MRZScanner()
```

After initializing the model, prepare an image for inference:

:::tip
You can use the test image provided by `MRZScanner`:

Download link: [**midv2020_test_mrz.jpg**](https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg)

<div align="center">
<figure style={{width: "50%"}}>
![test_mrz](./resources/test_mrz.jpg)
</figure>
</div>
:::

```python
import docsaidkit as D

img = D.imread('path/to/run_test_card.jpg')
```

Or you can read it directly from a URL:

```python
import cv2
from skimage import io

img = io.imread('https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
```

This image is quite long, and performing inference directly might cause excessive text deformation. Therefore, when calling the model, enable the `do_center_crop` parameter:

Next, you can perform inference using the `model`:

```python
result, msg = model(img, do_center_crop=True)
print(result)
# >>> ('PCAZEQAOARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#      'C946302620AZE6707297F23031072W12IMJ<<<<<<<40')
print(msg)
# >>> <ErrorCodes.NO_ERROR: 'No error.'>
```

:::tip
`MRZScanner` is encapsulated with `__call__`, so you can directly call the instance for inference.
:::

:::info
We have implemented an automatic model download feature. When you use `DocAligner` for the first time, the model will be downloaded automatically.
:::

## Using with `DocAligner`

Looking closely at the output results above, you might notice a few typos even after using `do_center_crop`.

Since we performed full-image scanning earlier, the model might misinterpret some text in the image.

To improve accuracy, we can use `DocAligner` to help align the MRZ region:

```python
import cv2
from docaligner import DocAligner  # Import DocAligner
from mrzscanner import MRZScanner
from skimage import io

model = MRZScanner()
doc_aligner = DocAligner()

img = io.imread(
    'https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

flat_img = doc_aligner(img).doc_flat_img  # Align the MRZ region
print(model(flat_img))
# >>> ('PCAZEQAQARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#      'C946302620AZE6707297F23031072W12IMJ<<<<<<<40')
```

After using `DocAligner`, there's no need to use the `do_center_crop` parameter.

Now, you can see that the output results are more accurate—the MRZ region of the image has been successfully recognized.

## Error Messages

To help users understand the reasons behind any errors, we return an error message field covering the following content:

```python
class ErrorCodes(Enum):
    NO_ERROR = 'No error.'
    INVALID_INPUT_FORMAT = 'Invalid input format.'
    POSTPROCESS_FAILED_LINE_COUNT = 'Postprocess failed, number of lines not 2 or 3.'
    POSTPROCESS_FAILED_TD1_LENGTH = 'Postprocess failed, length of lines not 30 when `doc_type` is TD1.'
    POSTPROCESS_FAILED_TD2_TD3_LENGTH = 'Postprocess failed, length of lines not 36 or 44 when `doc_type` is TD2 or TD3.'
```

This primarily serves to perform preliminary filtering of the output results. Errors that are immediately noticeable, such as incorrect string lengths or an incorrect number of lines, can be detected here.

## Check Digit

The check digit is a crucial part of the MRZ used to ensure data accuracy. It verifies the correctness of numbers to prevent data entry errors.

- The detailed operational process is described in [**References#Check Digit**](./reference#check-digit).

---

What we want to emphasize here is: **We do not provide a check digit calculation function.**

Apart from the standard methods, MRZs from different regions may redefine how the check digit is calculated. Providing a specific check digit calculation method might limit user flexibility.

Additionally, our goal is to train a model focused on MRZ recognition, where each output format is automatically determined by the model. If you want to apply a check digit, you must know the target format in advance. Otherwise, you'd have to calculate the check digit for every possible format, which is not our objective.

There are many other open-source projects that offer check digit calculation functions. For instance, [**Arg0s1080/mrz**](https://github.com/Arg0s1080/mrz) provides methods for calculating check digits. We recommend users to use such projects directly.
