---
sidebar_position: 3
---

# Enums

In OpenCV, the enumeration classes are numerous. To facilitate their use, we have organized some commonly used enumeration classes into DocsaidKit. These enumerations provide a clear and convenient way to reference common parameters and types, enhancing code readability and maintainability.

Most enumeration values are directly referenced from OpenCV's enums to ensure consistency. If you need other enum values, you can directly refer to OpenCV's enums.

## Overview of Enumeration Classes

- [**INTER**](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/enums.py#L12): Defines different types of image interpolation methods.
- [**ROTATE**](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/enums.py#L20): Defines image rotation angles.
- [**BORDER**](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/enums.py#L26): Defines the modes of border handling.
- [**MORPH**](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/enums.py#L35): Defines the shapes of the kernel for morphological operations.
- [**COLORSTR**](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/enums.py#L41): Defines color strings for terminal display.
- [**FORMATSTR**](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/enums.py#L60): Defines formatting strings.
- [**IMGTYP**](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/enums.py#L66): Defines supported image file types.

## docsaidkit.INTER

Used for image resizing or resampling to choose the interpolation method.

- `NEAREST`: Nearest neighbor interpolation.
- `BILINEAR`: Bilinear interpolation.
- `CUBIC`: Cubic interpolation.
- `AREA`: Area interpolation.
- `LANCZOS4`: Lanczos interpolation (using 4 Lanczos windows).

## docsaidkit.ROTATE

Specific angles for image rotation.

- `ROTATE_90`: Rotate the image 90 degrees clockwise.
- `ROTATE_180`: Rotate the image 180 degrees.
- `ROTATE_270`: Rotate the image 90 degrees counterclockwise.

## docsaidkit.BORDER

Ways to expand the image borders.

- `DEFAULT`: Default border handling method.
- `CONSTANT`: Constant border, filled with a specific color.
- `REFLECT`: Reflective border.
- `REFLECT_101`: Another type of reflective border.
- `REPLICATE`: Replicate the edge pixels of the border.
- `WRAP`: Wrap around border.

## docsaidkit.MORPH

Shapes of the structural element used in morphological filtering.

- `CROSS`: Cross-shaped.
- `RECT`: Rectangular.
- `ELLIPSE`: Elliptical.

## docsaidkit.COLORSTR

Color codes used for console output.

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

## docsaidkit.FORMATSTR

Options for text formatting.

- `BOLD`: Bold.
- `ITALIC`: Italic.
- `UNDERLINE`: Underlined.

## docsaidkit.IMGTYP

Supported image file types.

- `JPEG`: JPEG format image.
- `PNG`: PNG format image.