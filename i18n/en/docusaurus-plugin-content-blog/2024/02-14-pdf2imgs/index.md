---
slug: convert-pdf-to-images
title: Convert PDF to Images using Python
authors: Zephyr
tags: [Python, pdf2image]
image: /en/img/2024/0214.webp
description: Using open-source library pdf2image.
---

We often need to convert PDF files into image formats.

So here, we recommend a handy Python module: [pdf2image](https://github.com/Belval/pdf2image/tree/master), which can convert PDF files into PIL images.

<!-- truncate -->

## Install Dependencies

`pdf2image` relies on `pdftoppm` and `pdftocairo`, and installation varies slightly across different operating systems:

- **Mac**: Install Poppler via Homebrew: `brew install poppler`.
- **Linux**: Most Linux distributions come pre-installed with `pdftoppm` and `pdftocairo`. If not, install `poppler-utils` via your package manager.
- **Using `conda`**: Poppler can be installed via `conda` on any platform: `conda install -c conda-forge poppler`, then proceed to install `pdf2image`.

## Install `pdf2image`

First, you need to install `pdf2image`. Enter the following command in your terminal to install:

```shell
pip install pdf2image
```

## Convert PDF using `pdf2image`

Converting PDF to images is straightforward:

```python
from pdf2image import convert_from_path

images = convert_from_path('/path/to/your/pdf/file.pdf')
```

This will convert each page of the PDF into a PIL image object and store them in the `images` list.

You can also convert PDF from binary data:

```python
images = convert_from_bytes(open('/path/to/your/pdf/file.pdf', 'rb').read())
```

## Optional Parameters

`pdf2image` provides extensive optional parameters, allowing you to customize DPI, output format, page ranges, etc. For example: use `dpi=300` to enhance the clarity of the output images, or use `first_page` and `last_page` to specify the conversion range.

You can refer to the:

- [**official documentation**](https://github.com/Belval/pdf2image/tree/master) of `pdf2image`;

or check our own modified:

- [**pdf2imgs**](https://github.com/DocsaidLab/DocsaidKit/blob/eb8ac0a56779a75dcc951c683001e6129052cc5a/docsaidkit/vision/improc.py#L275)

function for more usage examples.

## Conclusion

`pdf2image` is a powerful and easy-to-use tool that meets your needs for converting PDF to images. Whether it's for document processing, data organization, or content presentation, it provides an efficient solution.
