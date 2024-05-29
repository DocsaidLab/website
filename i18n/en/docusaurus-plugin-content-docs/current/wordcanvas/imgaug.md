---
sidebar_position: 5
---

# Augmentation

We have not included image augmentation functionality directly within `WordCanvas` because we believe it is a highly "customizable" requirement. Different application scenarios may require different augmentation methods. However, we provide some simple examples to illustrate how to implement image augmentation.

We prefer using the [**albumentations**](https://github.com/albumentations-team/albumentations) package for image augmentation, but you can use any library you prefer.

## Example 1: Shear Transformation

After generating a text image, apply custom operations.

First, we demonstrate applying a shear transformation using `Shear`:

The `Shear` class is responsible for performing shear transformations on images. Shearing changes the geometric shape of an image, creating a horizontal slant, which can help models learn to recognize objects in different directions and positions.

- **Parameters**

  - max_shear_left: Maximum shear angle to the left. Default is 20 degrees.
  - max_shear_right: Maximum shear angle to the right. Default is also 20 degrees.
  - p: Probability of applying the operation. Default is 0.5, meaning any given image has a 50% chance of being sheared.

- **Usage**

  ```python
  from wordcanvas import Shear, WordCanvas

  gen = WordCanvas()
  shear = Shear(max_shear_left=20, max_shear_right=20, p=0.5)

  img, _ = gen('Hello, World!')
  img = shear(img)
  ```

  ![shear_example](./resources/shear_example.jpg)

## Example 2: Rotation Transformation

To implement rotation transformation, we use the `SafeRotate` class from `albumentations`.

When using operations like Shift, Scale, or Rotate, you might encounter background color filling issues.

In this case, you should call the `infos` information to get the background color.

```python
from wordcanvas import ExampleAug, WordCanvas
import albumentations as A

gen = WordCanvas(
    background_color=(255, 255, 0),
    text_color=(0, 0, 0)
)

aug =  A.SafeRotate(
    limit=30,
    border_mode=cv2.BORDER_CONSTANT,
    value=infos['background_color'],
    p=1
)

img, infos = gen('Hello, World!')
img = aug(image=img)['image']
```

![rotate_example](./resources/rotate_example.jpg)

## Example 3: Modifying Class Behavior

By now, you might notice:

- If each time `WordCanvas` generates an image with a random background color, it requires reinitializing the `albumentations` class every time, which might seem inefficient.

Perhaps we can modify the behavior of `albumentations` so that it can be used continuously after a single initialization?

```python
import albumentations as A
import cv2
import numpy as np
from wordcanvas import WordCanvas

gen = WordCanvas(
    random_background_color=True
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
    aug.value = infos['background_color']

    img = aug(image=img)['image']

    imgs.append(img)

# Display results
img = np.concatenate(imgs, axis=0)
```

![bgcolor_example](./resources/bgcolor_example.jpg)

:::danger
We still recommend using the method from Example 2 (even though it might seem a bit cumbersome) because if you directly modify the behavior of `albumentations` classes, it can cause issues in multi-threaded training environments. Please be careful!
:::

## Example 4: Adding Backgrounds

You might want more than just plain text images and want to add backgrounds to improve the generalization ability of your model.

In this case, you need to prepare a set of background images and then refer to the following example:

```python
import albumentations as A
import cv2
import numpy as np
from wordcanvas import WordCanvas
from albumentations import RandomCrop

gen = WordCanvas(
    random_text_color=True,
    random_background_color=True
)

# Generate a random color text image
img, infos = gen('Hello, World!')
```

![sample25](./resources/sample25.jpg)

Next, load a background image:

```python
bg = cv2.imread('path/to/your/background.jpg')
```

[![bg_example](./resources/bg_example.jpg)](https://www.lccnet.com.tw/lccnet/article/details/2274)

Finally, randomly crop a region from the background and place the text image on it:

```python
bg = RandomCrop(img.shape[0], img.shape[1])(image=bg)['image']

result_img = np.where(img == infos['background_color'], bg, img)
```

![bg_result](./resources/sample26.jpg)

## Example 5: Perspective Transformation

Perspective transformation is a technique that projects an image onto a new viewpoint, simulating the appearance of objects from different angles and distances.

Continuing from the previous example, apply a perspective transformation to the image before overlaying the background:

```python
from albumentations import Perspective

aug = A.Perspective(
    keep_size=True,
    fit_output=True,
    pad_val=infos['background_color'],
)

img = aug(image=img)['image']
result_img = np.where(img == infos['background_color'], bg, img)
```

![sample27](./resources/sample27.jpg)

:::tip
For spatial transformation image augmentations, we recommend applying perspective transformation to the original image first, then overlaying the background image. This way, the background image won't have strange black edges.
:::

## Example 6: Strong Light Reflection

Text images are also prone to issues with strong light reflections. In this case, we can use `RandomSunFlare` to simulate this condition:

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
For pixel-level image augmentations, we recommend overlaying the background image first, then applying image augmentation transformations. This way, you won't lose background information and avoid messy spots.
:::

## Conclusion

This concludes the introduction to this project. If you have any questions or suggestions, feel free to leave a comment below, and we will respond as soon as possible.

Alternatively, if you're unsure how to implement a specific operation, feel free to leave a comment as well, and we will do our best to assist you.

Happy using!
