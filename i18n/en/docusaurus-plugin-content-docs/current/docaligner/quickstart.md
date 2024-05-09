---
sidebar_position: 3
---

# QuickStart

We provide a simple model inference interface that includes pre-processing and post-processing logic.

First, you need to import the necessary dependencies and create a `DocAligner` class.

## Model Inference

Here is a simple example demonstrating how to use `DocAligner` for model inference:

```python
from docaligner import DocAligner

model = DocAligner()
```

After initializing the model, you'll need to prepare an image for inference:

:::tip
You can use a test image provided by `DocAligner`:

Download link: [**run_test_card.jpg**](https://github.com/DocsaidLab/DocAligner/blob/main/docs/run_test_card.jpg)
:::

```python
import docsaidkit as D

img = D.imread('path/to/run_test_card.jpg')
```

Or you can directly read it from a URL:

```python
import cv2
from skimage import io

img = io.imread('https://github.com/DocsaidLab/DocAligner/blob/main/docs/run_test_card.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
```

![test_card](./resources/run_test_card.jpg)

Next, you can use the `model` for inference:

```python
result = model(img)
```

The inference result you receive is packaged as a [**Document**](../docsaidkit/funcs/objects/document) type, containing the document's polygon, OCR text information, etc. In this module, we won't use the OCR features, so we will only utilize the `image` and `doc_polygon` attributes. After obtaining the inference result, you can perform various post-processing operations.

:::tip
`DocAligner` has encapsulated the inference using the `__call__` method, so you can directly call the instance for inference.
:::

:::tip
**Model Download**: We have designed an automatic model download feature. When you use `DocAligner` for the first time, the model will be downloaded automatically.
:::

## Output Results

### 1. Drawing the Polygon

Draw and save an image with the document polygon.

```python
# draw
result.draw_doc(
    folder='path/to/save/folder',
    name='output_image.jpg'
)
```

Or without specifying a path, directly output:

```python
# The default output path is the current directory
# The default file name uses the current time, as "output_{D.now()}.jpg".
result.draw_doc()
```

![output_image](./resources/flat_result.jpg)

### 2. Obtaining the NumPy Image

If you have other needs, you can use the `gen_doc_info_image` method and then process it as needed.

```python
img = result.gen_doc_info_image()
```

### 3. Extracting the Flattened Image

If you know the original size of the document, you can use the `gen_doc_flat_img` method to transform the document image from its polygonal boundary into a rectangular image.

```python
H, W = 1080, 1920
flat_img = result.gen_doc_flat_img(image_size=(H, W))
```

If the image class is unknown, you can omit the `image_size` parameter. In this case, the "smallest rectangle" will be automatically calculated based on the document polygon's boundary, setting the rectangle's dimensions as `H` and `W`.

```python
flat_img = result.gen_doc_flat_img()
```

:::tip
When your document is significantly skewed in the image, calculating a more flattened smallest rectangle may occur, resulting in some distortion upon flattening. Therefore, it's recommended to manually set the `image_size` parameter in such cases.
:::
