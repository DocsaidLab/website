---
sidebar_position: 4
---

# Advanced

Beyond basic usage, `WordCanvas` offers several advanced settings that allow you to flexibly control the output of text images. Here we introduce randomness settings, which are primarily used for training models.

## Random Fonts

Enable the random font feature using the `random_font` parameter. When `random_font` is set to `True`, the `font_bank` parameter becomes active, while `font_path` is ignored.

You should specify the `font_bank` parameter to your font library since the default is the package's `fonts` directory. For demonstration, we've placed two fonts in the `fonts` directory, so if you do not modify `font_bank`, it will randomly select from these two fonts.

```python
import numpy as np
from wordcanvas import WordCanvas

gen = WordCanvas(
    random_font=True,
    output_size=(64, 512),
    font_bank="path/to/your/font/bank"
)

imgs = []
for _ in range(8):
    text = 'Hello, World!'
    img, infos = gen(text)
    imgs.append(img)

# Combine all images for output
img = np.concatenate(imgs, axis=0)
```

![sample17](./resources/sample17.jpg)

## Random Text Content

If you are unsure of what text to generate, you can use the `random_text` parameter.

When `random_text` is set to `True`, the originally input `text` will be ignored.

```python
import numpy as np
from wordcanvas import WordCanvas

gen = WordCanvas(
    random_text=True,
    output_size=(64, 512),
)

imgs = []
for _ in range(8):
    text = 'Hello!World!' # This input will be ignored
    img, infos = gen(text)
    imgs.append(img)

# Combine all images for output
img = np.concatenate(imgs, axis=0)
```

![sample18](./resources/sample18.jpg)

## Specifying Text Length

When `random_text` is enabled, you can use:

- `min_random_text_length`: Minimum text length
- `max_random_text_length`: Maximum text length

These two parameters specify the range of text lengths.

```python
import numpy as np
from wordcanvas import WordCanvas

# Always generate text with 5 characters
gen = WordCanvas(
    random_text=True,
    min_random_text_length=5,
    max_random_text_length=5,
    output_size=(64, 512),
)

imgs = []
for _ in range(8):
    img, infos = gen()
    imgs.append(img)

# Combine all images for output
img = np.concatenate(imgs, axis=0)
```

![sample19](./resources/sample19.jpg)

## Random Background Color

Use the `random_background_color` parameter to enable the random background color feature.

```python
import numpy as np
from wordcanvas import WordCanvas

gen = WordCanvas(
    random_background_color=True,
    output_size=(64, 512),
)

imgs = []
for _ in range(8):
    text = 'Hello, World!'
    img, infos = gen(text)
    imgs.append(img)

# Combine all images for output
img = np.concatenate(imgs, axis=0)
```

![sample20](./resources/sample20.jpg)

## Random Text Color

Use the `random_text_color` parameter to enable the random text color feature.

```python
import numpy as np
from wordcanvas import WordCanvas

gen = WordCanvas(
    random_text_color=True,
    output_size=(64, 512),
)

imgs = []
for _ in range(8):
    text = 'Hello, World!'
    img, infos = gen(text)
    imgs.append(img)

# Combine all images for output
img = np.concatenate(imgs, axis=0)
```

![sample21](./resources/sample21.jpg)

## Random Text Alignment

Use the `random_align_mode` parameter to enable the random text alignment feature.

```python
import numpy as np
from wordcanvas import WordCanvas

gen = WordCanvas(
    random_align_mode=True,
    output_size=(64, 512),
)

imgs = []
for _ in range(8):
    text = 'Hello, World!'
    img, infos = gen(text)
    imgs.append(img)

# Combine all images for output
img = np.concatenate(imgs, axis=0)
```
![sample22](./resources/sample22.jpg)

## Random Text Direction

Use the `random_direction` parameter to enable the random text direction feature.

It's recommended to use this parameter in conjunction with `output_direction` for convenient image output.

```python
import numpy as np
from wordcanvas import WordCanvas, OutputDirection

gen = WordCanvas(
    random_direction=True,
    output_direction=OutputDirection.Horizontal,
    output_size=(64, 512),
)

imgs = []
for _ in range(8):
    text = '午安，或是晚安。'
    img, infos = gen(text)
    imgs.append(img)

# Combine all images for output
img = np.concatenate(imgs, axis=0)
```

![sample23](./resources/sample23.jpg)

## Full Randomization

If you want all settings to be random, you can use the `enable_all_random` parameter.

This parameter activates a mode where everything is randomized.

```python
import numpy as np
from wordcanvas import WordCanvas

gen = WordCanvas(
    enable_all_random=True,
    output_size=(64, 512),
)

imgs = []
for _ in range(20):
    img, infos = gen()
    imgs.append(img)

# Combine all images for output
img = np.concatenate(imgs, axis=0)
```

![sample24](./resources/sample24.jpg)

:::warning
This parameter does not adjust parameters that require reinitialization, such as `random_font`, `random_text`, etc. These need to be set manually.
:::

## Dashboard Revisited

Let's return to the dashboard feature.

![dashboard](./resources/dashboard.jpg)

When randomness-related parameters are enabled, parameters set to True will be marked in green, while those set to False will be marked in red.

We hope this design allows you to quickly verify related settings.

## Conclusion

During the development of this tool, our goal was to create a versatile tool capable of generating a variety of text images, particularly for training machine learning models. Introducing randomness aims to simulate various real-world scenarios, which is crucial for enhancing the adaptability and generalization ability of models.

This concludes the advanced usage section of this project, and we hope you find these features helpful for your applications.