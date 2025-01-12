---
sidebar_position: 5
---

# Augmentation

We did not implement the image augmentation feature within `WordCanvas` because we consider it a highly "customized" requirement. Different use cases may require different augmentation methods. However, we provide some simple examples to demonstrate how the image augmentation process can be implemented.

We typically use the [**albumentations**](https://github.com/albumentations-team/albumentations) library for image augmentation, but you are free to use any library of your choice.

:::info
After `albumentations` was updated to v2.0.0, many operation parameter names have changed. Please be aware of this.

For more details, refer to: [**albumentations v2.0.0**](https://github.com/albumentations-team/albumentations/releases/tag/2.0.0)
:::

## Example 1: Shear Transformation

After generating the text image, apply custom operations.

Here, we demonstrate applying a shear transformation using `Shear`:

The `Shear` class is responsible for applying a shear transformation to the image. Shearing alters the geometric shape of the image, creating a horizontal tilt, which can help the model learn to recognize objects at different angles and positions.

- **Parameters**

  - max_shear_left: Maximum shear angle to the left. The default is 20 degrees.
  - max_shear_right: Maximum shear angle to the right. The default is also 20 degrees.
  - p: Probability of applying the operation. The default is 0.5, meaning there’s a 50% chance that any given image will be sheared.

- **Usage**

  ```python
  from wordcanvas import Shear, WordCanvas

  gen = WordCanvas()
  shear = Shear(max_shear_left=20, max_shear_right=20, p=0.5)

  img = gen('Hello, World!')
  img = shear(img)
  ```

  ![shear_example](./resources/shear_example.jpg)

## Example 2: Rotation Transformation

To implement rotation transformation, we import the `SafeRotate` class from `albumentations`.

When using operations like `Shift`, `Scale`, or `Rotate`, issues related to background color filling may arise.

In this case, you should call `infos` to obtain the background color.

```python
import cv2
from wordcanvas import ExampleAug, WordCanvas
import albumentations as A

gen = WordCanvas(
    background_color=(255, 255, 0),
    text_color=(0, 0, 0),
    return_infos=True
)

img, infos = gen('Hello, World!')

aug = A.SafeRotate(
    limit=30,
    border_mode=cv2.BORDER_CONSTANT,
    fill=infos['background_color'],
    p=1
)

img = aug(image=img)['image']
```

![rotate_example](./resources/rotate_example.jpg)

## Example 3: Modify Class Behavior

At this point in the code, you might notice:

- If each image generated has a random background color, then `albumentations` needs to be reinitialized every time, which doesn’t seem efficient.

Perhaps we can modify the behavior of `albumentations` so that it only needs to be initialized once and can be reused?

```python
import albumentations as A
import cv2
import numpy as np
from wordcanvas import RandomWordCanvas


gen = RandomWordCanvas(
    random_background_color=True,
    return_infos=True
)

aug = A.SafeRotate(
    limit=30,
    border_mode=cv2.BORDER_CONSTANT,
    p=1
)

imgs = []
for _ in range(8):
    img, infos = gen('Hello, World!')

    # Modify albu class behavior
    aug.fill = infos['background_color']

    img = aug(image=img)['image']

    imgs.append(img)

# Display results
img = np.concatenate(imgs, axis=0)
```

![bgcolor_example](./resources/bgcolor_example.jpg)

:::danger
We still recommend using the method from Example 2 (even though it may seem inefficient), as modifying `albumentations`' class behavior could cause issues in multi-threaded training environments. Please be cautious!
:::

## Example 4: Adding Background

If you’re not satisfied with a simple text image and want to add a background to enhance the model's generalization ability, you will need to prepare a set of background images and follow the example below:

```python
import albumentations as A
import cv2
import numpy as np
from wordcanvas import RandomWordCanvas
from albumentations import RandomCrop

gen = RandomWordCanvas(
    random_text_color=True,
    random_background_color=True,
    return_infos=True
)

# Generate a random text image
img, infos = gen('Hello, World!')
```

![sample25](./resources/sample25.jpg)

Then, load a background image:

```python
bg = cv2.imread('path/to/your/background.jpg')
```

[![bg_example](./resources/bg_example.jpg)](https://www.lccnet.com.tw/lccnet/article/details/2274)

Finally, crop a random region from the background and place the text image on top:

```python
bg = RandomCrop(img.shape[0], img.shape[1])(image=bg)['image']

result_img = np.where(img == infos['background_color'], bg, img)
```

![bg_result](./resources/sample26.jpg)

## Example 5: Perspective Transformation

Perspective transformation projects an image onto a new viewplane. This type of transformation can simulate how objects appear from different angles and distances.

We continue with the previous example and apply a perspective transformation to the image before adding the background:

```python
from albumentations import Perspective

aug = A.Perspective(
    keep_size=True,
    fit_output=True,
    fill=infos['background_color'],
)

img = aug(image=img)['image']
result_img = np.where(img == infos['background_color'], bg, img)
```

![sample27](./resources/sample27.jpg)

:::tip
For "spatial transformation" augmentation operations, we recommend performing the perspective transformation first, followed by adding the background image. This ensures that the background won’t have strange black edges.
:::

## Example 6: Sun Flare

Text images often have issues with strong light reflections. In this case, we can use `RandomSunFlare` to simulate this effect:

```python
from albumentations import RandomSunFlare

aug = A.RandomSunFlare(
    src_radius=128,
    src_color=(255, 255, 255),
)

result_img = aug(image=result_img)['image']
```

![sample28](./resources/sample28.jpg)

:::tip
For "pixel modification" augmentation operations, we recommend adding the background image first and then applying the image transformation. This prevents background information from being lost, which could result in random noise.
:::

## Conclusion

This concludes the introduction to the project. If you have any questions or suggestions, feel free to leave a comment below, and we will reply as soon as possible.

If you’re unsure how to implement a certain operation, you are also welcome to leave a comment. We will do our best to assist you.

Enjoy using it!
