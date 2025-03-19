---
slug: convert-pdf-to-images
title: Convert PDF to Images with Python
authors: Z. Yuan
tags: [Python, pdf2image]
image: /en/img/2024/0214.webp
description: Solve the problem with the open-source pdf2image package.
---

During development, you may often need to convert PDF files into image formats, whether for document display, data processing, or content sharing.

This article introduces a handy Python module: [**pdf2image**](https://github.com/Belval/pdf2image/tree/master), which can convert PDF files into PIL images.

<!-- truncate -->

## Install Dependencies

`pdf2image` depends on two tools: `pdftoppm` and `pdftocairo`, with different installation methods depending on the operating system:

- **Mac**: Install Poppler via Homebrew by running the following in the terminal:

  ```shell
  brew install poppler
  ```
- **Linux**: Most Linux distributions have `pdftoppm` and `pdftocairo` pre-installed. If not, you can install them with the following command:

  ```shell
  sudo apt-get install poppler-utils   # For Ubuntu/Debian systems
  ```
- **Using `conda`**: You can install Poppler via `conda` on any platform:

  ```shell
  conda install -c conda-forge poppler
  ```

  Once installed, you can install `pdf2image`.

## Install `pdf2image`

To install, run the following command in the terminal:

```shell
pip install pdf2image
```

## Usage

The basic usage for converting a PDF to an image is quite simple.

Here’s an example of how to convert each page of a PDF into a PIL image object and save it as a file:

```python
from pdf2image import convert_from_path

# Convert PDF to a list of images
images = convert_from_path('/path/to/your/pdf/file.pdf')

# Save each page as a PNG image
for i, image in enumerate(images):
    image.save(f'output_page_{i+1}.png', 'PNG')
```

If you want to convert from binary data, you can do it as follows:

```python
with open('/path/to/your/pdf/file.pdf', 'rb') as f:
    pdf_data = f.read()

images = convert_from_bytes(pdf_data)
```

## Optional Parameters and Advanced Settings

`pdf2image` provides rich optional parameters that allow you to customize the quality and range of the output images:

- **DPI Setting**: Adjusting the `dpi` parameter can increase the image resolution, suitable for cases where high-quality images are required:

  ```python
  images = convert_from_path('/path/to/your/pdf/file.pdf', dpi=300)
  ```

- **Specify Page Range**: Use the `first_page` and `last_page` parameters to convert only specific pages:

  ```python
  images = convert_from_path('/path/to/your/pdf/file.pdf', first_page=2, last_page=5)
  ```

- **Output Image Format**: The `fmt` parameter allows you to specify the output image format, such as JPEG or PNG:

  ```python
  images = convert_from_path('/path/to/your/pdf/file.pdf', fmt='jpeg')
  ```

- **Error Handling**: During the conversion process, you might encounter format errors or corrupted files. It’s recommended to use try/except to catch exceptions:

  ```python
  try:
      images = convert_from_path('/path/to/your/pdf/file.pdf')
  except Exception as e:
      print("Conversion failed:", e)
  ```

## Conclusion

`pdf2image` is a useful tool. For more parameters and detailed usage, refer to the [**official pdf2image documentation**](https://github.com/Belval/pdf2image/tree/master).