# Enums

There are too many enumeration classes in OpenCV. To make it easier to use, we have organized some common enumeration classes into `capybara`. These enumeration classes provide a clear and convenient way to reference commonly used parameters and types, which helps improve the readability and maintainability of the code.

Most of the enumeration values directly reference OpenCV’s enumeration values, ensuring consistency. If you need to use other enumeration values, you can directly reference OpenCV’s enumeration values.

## Overview of Enumeration Classes

- [**INTER**](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/enums.py#L12): Defines different image interpolation methods.
- [**ROTATE**](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/enums.py#L20): Defines the rotation angles of an image.
- [**BORDER**](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/enums.py#L26): Defines border handling methods.
- [**MORPH**](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/enums.py#L35): Defines the kernel shapes for morphological operations.
- [**COLORSTR**](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/enums.py#L41): Defines color strings for terminal display.
- [**FORMATSTR**](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/enums.py#L60): Defines formatting strings for text.
- [**IMGTYP**](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/enums.py#L66): Defines supported image file types.

## capybara.INTER

Interpolation methods used for image resizing or resampling.

- `NEAREST`: Nearest neighbor interpolation.
- `BILINEAR`: Bilinear interpolation.
- `CUBIC`: Cubic interpolation.
- `AREA`: Area-based interpolation.
- `LANCZOS4`: Lanczos interpolation (using 4 Lanczos windows).

## capybara.ROTATE

Specific angles for image rotation.

- `ROTATE_90`: Rotate 90 degrees clockwise.
- `ROTATE_180`: Rotate 180 degrees.
- `ROTATE_270`: Rotate 90 degrees counterclockwise.

## capybara.BORDER

Methods for extending image borders.

- `DEFAULT`: Default border handling.
- `CONSTANT`: Constant border, filled with a specific color.
- `REFLECT`: Reflective border.
- `REFLECT_101`: Another type of reflective border.
- `REPLICATE`: Replicate the edge pixels.
- `WRAP`: Wrap-around border.

## capybara.MORPH

Shapes of the structural elements used in morphological filtering.

- `CROSS`: Cross-shaped.
- `RECT`: Rectangular.
- `ELLIPSE`: Elliptical.

## capybara.COLORSTR

Color codes for console output.

- `BLACK`: Black.
- `RED`: Red.
- `GREEN`: Green.
- `YELLOW`: Yellow.
- `BLUE`: Blue.
- `MAGENTA`: Magenta.
- `CYAN`: Cyan.
- `WHITE`: White.
- `BRIGHT_BLACK`: Bright black.
- `BRIGHT_RED`: Bright red.
- `BRIGHT_GREEN`: Bright green.
- `BRIGHT_YELLOW`: Bright yellow.
- `BRIGHT_BLUE`: Bright blue.
- `BRIGHT_MAGENTA`: Bright magenta.
- `BRIGHT_CYAN`: Bright cyan.
- `BRIGHT_WHITE`: Bright white.

## capybara.FORMATSTR

Text formatting options.

- `BOLD`: Bold.
- `ITALIC`: Italic.
- `UNDERLINE`: Underline.

## capybara.IMGTYP

Supported image file types.

- `JPEG`: JPEG format image.
- `PNG`: PNG format image.
