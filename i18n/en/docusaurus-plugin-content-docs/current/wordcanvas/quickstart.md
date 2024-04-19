---
sidebar_position: 3
---

# QuickStart

Getting started is often the hardest part, so let's keep it simple.

## Starting with a String

Start with a basic declaration to begin using the tool.

```python
from wordcanvas import WordCanvas

gen = WordCanvas()
```

Using default settings, you can directly call the function to generate a text image.

```python
text = "你好！Hello, World!"
img, infos = gen(text)
print(img.shape)
# >>> (67, 579, 3)
```

![sample1](./resources/sample1.jpg)

:::tip
In default mode, the output image size depends on:

1. **Font Size**: Default is 64, which increases the image size as the font size increases.
2. **Text Length**: The longer the text, the wider the image, with the specific length determined by `pillow`.
:::

## Specifying a Specific Font

You can specify your preferred font using the `font` parameter.

```python
gen = WordCanvas(
    font_path="/path/to/your/font/OcrB-Regular.ttf"
)

text = 'Hello, World!'
img, infos = gen(text)
```
![sample14](./resources/sample14.jpg)

When the font does not support the input text, tofu characters will appear.

```python
text = 'Hello, 中文!'
img, infos = gen(text)
```

![sample15](./resources/sample15.jpg)

:::tip
**Method to check if the font supports the characters:**

Currently, I don't need this, so I left a basic method here. This method is a simple check, which can only check one character at a time, so it requires iterating over all characters. If you have other requirements, please expand it yourself.

```python title="check_font.py"
from wordcanvas import is_character_supported, load_ttfont

target_text = 'Hello, 中文!'

font = load_ttfont("/path/to/your/font/OcrB-Regular.ttf")

for c in target_text:
    status = is_character_supported(font, c)
    if not status:
        print(f"Character: {c}, Not Supported!")

# >>> Character: 中, Not Supported!
# >>> Character: 文, Not Supported!
```
:::

## Setting Image Size

Use the `output_size` parameter to adjust the image size.

```python
gen = WordCanvas(output_size=(64, 1024)) # Height 64, Width 1024
img, infos = gen(text)
print(img.shape)
# >>> (64, 1024, 3)
```

![sample4](./resources/sample4.jpg)

When the set size is smaller than the text image size, the text image will automatically be scaled down.

That is, the text will be squeezed together, forming a thin rectangle, like this:

```python
text = '你好' * 10
gen = WordCanvas(output_size=(64, 512))  # Height 64, Width 512
img, infos = gen(text)
```

![sample8](./resources/sample8.jpg)

## Adjusting Background Color

Use the `background_color` parameter to adjust the background color.

```python
gen = WordCanvas(background_color=(255, 0, 0)) # Red background
img, infos = gen(text)
```

![sample2](./resources/sample2.jpg)

## Adjusting Text Color

Use the `text_color` parameter to adjust the text color.

```python
gen = WordCanvas(text_color=(0, 255, 0)) # Green text
img, infos = gen(text)
```

![sample3](./resources/sample3.jpg)

## Adjusting Text Alignment

:::warning
Remember the image size we mentioned earlier? In default settings, **setting text alignment is meaningless**. You must allow extra space in the text image to see the effect of alignment.
:::

Use the `align_mode` parameter to adjust the text alignment mode.

```python
from wordcanvas import AlignMode, WordCanvas

gen = WordCanvas(
    output_size=(64, 1024),
    align_mode=AlignMode.Center
)

text = '你好！ Hello, World!'
img, infos = gen(text)
```

- **Center alignment: `AlignMode.Center`**

    ![sample5](./resources/sample5.jpg)

- **Right alignment: `AlignMode.Right`**

    ![sample6](./resources/sample6.jpg)

- **Left alignment: `AlignMode.Left`**

    ![sample7](./resources/sample4.jpg)

- **Scatter alignment: `AlignMode.Scatter`**

    ![sample8](./resources/sample7.jpg)

    :::tip
    In scatter alignment mode, not every character is spread out; it is done by word units. In Chinese, the unit is one character; in English, the unit is one space.

    As shown in the above image, the input text "你好！ Hello, World!" is split into:

    - ["你", "好", "！", "Hello,", "World!"]

    After ignoring spaces, they are then aligned scatteredly.

    Also, when the input text can only be split into one word, Chinese word scatter alignment equates to center alignment, and English words are split into characters before being scatteredly aligned.

    We use the following logic for this:

    ```python
    def split_text(text: str):
        """ Split text into a list of characters. """
        pattern = r"[a-zA-Z0-9\p{P}\p{S}]+|."
        matches = regex.findall(pattern, text)
        matches = [m for m in matches if not regex.match(r'\p{Z}', m)]
        if len(matches) == 1:
            matches = list(text)
        return matches
    ```

    :::warning
    This is a very simple implementation and may not meet all needs. If you have a more complete solution for splitting strings, you are welcome to contribute.
    :::

## Adjusting Text Direction

Use the `direction` parameter to adjust the text direction.

- **Outputting horizontal text**

    ```python
    text = '你好！'
    gen = WordCanvas(direction='ltr') # Left to right horizontal text
    img, infos = gen(text)
    ```

    ![sample9](./resources/sample9.jpg)

- **Outputting vertical text**

    ```python
    text = '你好！'
    gen = WordCanvas(direction='ttb') # Top to bottom vertical text
    img, infos = gen(text)
    ```

    ![sample10](./resources/sample10.jpg)

- **Outputting vertical text with scatter alignment**

    ```python
    text = '你好！'
    gen = WordCanvas(
        direction='ttb',
        align_mode=AlignMode.Scatter,
        output_size=(64, 512)
    )
    img, infos = gen(text)
    ```

    ![sample11](./resources/sample11.jpg)

## Adjusting Output Direction

Use the `output_direction` parameter to adjust the output direction.

:::tip
**The appropriate time to use this parameter is** when you choose to output vertical text but wish to view the text image horizontally.
:::

- **Vertical text, horizontal output**

    ```python
    from wordcanvas import OutputDirection, WordCanvas

    gen = WordCanvas(
        direction='ttb',
        output_direction=OutputDirection.Horizontal
    )

    text = '你好！'
    img, infos = gen(text)
    ```

    ![sample12](./resources/sample12.jpg)

- **Horizontal text, vertical output**

    ```python
    from wordcanvas import OutputDirection, WordCanvas

    gen = WordCanvas(
        direction='ltr',
        output_direction=OutputDirection.Vertical
    )

    text = '你好！'
    img, infos = gen(text)
    ```

    ![sample13](./resources/sample13.jpg)

## Flattening Text

In scenarios where the text is particularly flat, you can use the `text_aspect_ratio` parameter.

```python
gen = WordCanvas(
    text_aspect_ratio=0.25, # Text height / text width = 1/4
    output_size=(32, 1024),
)  # Flattened text

text = "Flattened test"
img, infos = gen(text)
```

![sample16](./resources/sample16.jpg)

:::info
Note that when the flattened text size exceeds the `output_size`, the image will undergo automatic scaling. Therefore, even though you flattened the image, it might be scaled back, resulting in no apparent change.
:::

## Dashboard

That's a brief overview of the basic functionality.

Finally, let's take a look at the dashboard feature.

```python
gen = WordCanvas()
print(gen)
```

You can also skip `print` and just output directly, as we've implemented the `__repr__` method.

The output will display a simple dashboard.

![dashboard](./resources/dashboard.jpg)

You can see:

- The first column is Property, which lists all the settings.
- The second column is Current Value, which shows the value of the parameters "at this moment."
- The third column is SetMethod, which describes the method to set the parameter. Parameters marked `set` can be directly modified; those marked `reinit` require reinitialization of the `WordCanvas` object.
- The fourth column is DType, which is the data type of the parameter.
- The fifth column is Description, which describes the parameter.

Most parameters can be directly set, meaning when you need to change output characteristics, you don't need to rebuild a `WordCanvas` object, just set them directly. Parameters that require `reinit` typically involve font initialization, like `text_size`. So, be aware, not all parameters can be directly set.

```python
gen.output_size = (64, 1024)
gen.text_color = (0, 255, 0)
gen.align_mode = AlignMode.Center
gen.direction = 'ltr'
gen.output_direction = OutputDirection.Horizontal
```

After setting, simply call to get the new text image.

If you've set a parameter that requires `reinit`, you'll encounter an error:

-  **AttributeError: can't set attribute**

    ```python
    gen.text_size = 128
    # >>> AttributeError: can't set attribute
    ```

:::danger
Of course, you can still forcibly set parameters, and as a fellow Python user, I can't stop you:

```python
gen._text_size = 128
```

But doing this will cause errors later on!

Don't insist, just reinitialize an object instead.
:::

## Summary

While many features weren't mentioned, this covers the basic functionalities.

That concludes the basic usage of this project; the next chapter will introduce advanced features.