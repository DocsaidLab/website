---
sidebar_position: 3
---

# QuickStart

We provide a simple model inference interface that includes logic for preprocessing and postprocessing.

First, you need to import the relevant dependencies and create a `DocClassifier` class.

## Data Registration

Before diving into the model, let's talk about data registration.

---

In the inference folder directory, there is a `register` folder containing all the registration data. You can place your registration data in this folder, and `DocClassifier` will automatically read all the data in the folder during inference. If you want to use your own dataset, specify the `register_root` parameter when creating the `DocClassifier` and set it to the root directory of your dataset.

We have included several registered document images in the module for demonstration purposes, which you can refer to and expand upon. Additionally, we strongly recommend using your own dataset to ensure the model can adapt to your application scenarios.

![register](./resources/register_demo.jpg)

:::tip
We recommend using full-size images for data, minimizing background interference to improve model stability.
:::

:::danger
Many of the pre-registered images in the folder are collected from the internet and have very low resolutions, suitable only for demonstration purposes and not for deployment. Please make good use of the `register_root` parameter with your own dataset to ensure the model can adapt to your application scenarios.
:::

## Duplicate Registration

This issue can occur in two scenarios:

- **Scenario 1: Duplicate File Names**

    In our implementation logic, the file names in the registration folder serve as the query index for the data.

    Therefore, when file names are duplicated, the later files will overwrite the earlier ones.

    This scenario is not problematic as the overwritten files will simply not be used, which does not affect model inference.

- **Scenario 2: Duplicate File Contents**

    Identical files are registered more than once.

    Suppose a user registers three identical images but with different labels. During inference, the scores in the similarity ranking process will be the same, but one will always be ranked first. In such cases, the model cannot guarantee returning the same label each time.

    This scenario introduces uncertainty into the model, making it difficult to determine the reasons behind the inconsistencies.

:::tip
In any case, please treat data registration seriously.
:::

## Model Inference

Once the registration data is prepared, we can start performing model inference.

Here's a simple example. First, initialize the model:

```python
from docclassifier import DocClassifier

model = DocClassifier()
```

Next, load an image:

```python
import docsaidkit as D

img = D.imread('path/to/test_driver.jpg')
```

:::tip
You can use the test image provided by `DocClassifier`:

Download link: [**test_driver.jpg**](https://github.com/DocsaidLab/DocClassifier/blob/main/docs/test_driver.jpg)

![test_card](./resources/test_driver.jpg)

:::

Or directly load it from a URL:

```python
import cv2
from skimage import io

img = io.imread('https://github.com/DocsaidLab/DocClassifier/blob/main/docs/test_driver.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
```

Finally, use the `model` for inference:

```python
most_similar, max_score = model(img)
print(f'most_similar: {most_similar}, max_score: {max_score:.4f}')
# >>> most_similar: None, max_score: 0.0000
```

In the default case, this example will return `None` and `0.0000` because the default registration data is very different from the input image. Therefore, the model determines a very low similarity between this image and the registration data.

:::tip
The `DocClassifier` is encapsulated with `__call__`, so you can directly call the instance for inference.
:::

:::info
We have implemented automatic model downloading. The model will be automatically downloaded the first time you use `DocClassifier`.
:::

## Threshold Setting

We evaluate the model's performance based on the TPR@FPR=1e-4 standard. However, this standard is relatively strict and may lead to poor user experience in deployment. Therefore, we suggest using thresholds based on TPR@FPR=1e-1 or TPR@FPR=1e-2 during deployment.

Currently, the default threshold is set using the `TPR@FPR=1e-2` standard, which we have determined to be more suitable based on our testing and evaluation. The detailed threshold settings are as follows:

- **lcnet050_cosface_f256_r128_squeeze_imagenet_clip_20240326 results**

    - **Setting `model_cfg` to "20240326"**
    - **TPR@FPR=1e-4: 0.912**

        |    FPR    |  1e-05  |  1e-04  |  1e-03  |  1e-02  |  1e-01  |   1     |
        | :-------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
        |    TPR    |  0.856  |  0.912  |  0.953  |  0.980  |  0.996  |   1.0   |
        | Threshold |  0.705  |  0.682  |  0.657  |  0.626  |  0.581  |  0.359  |
