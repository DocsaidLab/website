---
sidebar_position: 2
---

# Installation

Before you begin installing DocsaidKit, ensure your system meets the following requirements:

## Prerequisites

### Python Version

- Ensure that Python 3.8 or above is installed on your system.

### Dependency Packages

Install the required dependency packages according to your operating system.

- **Ubuntu**

  Open the terminal and run the following command to install the dependencies:

  ```bash
  sudo apt install libturbojpeg exiftool ffmpeg libheif-dev
  ```

- **MacOS**

  Use brew to install the dependencies:

  ```bash
  brew install jpeg-turbo exiftool ffmpeg libheif
  ```

### pdf2image Dependencies

pdf2image is a Python module for converting PDF documents into images.

Follow the instructions below to install the necessary dependencies based on your operating system:

- For detailed installation instructions, refer to the open-source project [**pdf2image**](https://github.com/Belval/pdf2image).

- MacOS: Mac users need to install poppler. Install it via Brew:

  ```bash
  brew install poppler
  ```

- Linux: Most Linux distributions come with `pdftoppm` and `pdftocairo` pre-installed.

  If not installed, use your package manager to install poppler-utils.

  ```bash
  sudo apt install poppler-utils
  ```

## Package Installation

Once the prerequisites are met, you can proceed with the installation via git clone:

1. Download the package:

   ```bash
   git clone https://github.com/DocsaidLab/DocsaidKit.git
   ```

2. Install the wheel package:

   ```bash
   pip install wheel
   ```

3. Build the wheel file:

   ```bash
   cd DocsaidKit
   python setup.py bdist_wheel
   ```

4. Install the built wheel package:

   ```bash
   pip install dist/docsaidkit-*-py3-none-any.whl
   ```

   If you need to install the version supporting PyTorch:

   ```bash
   pip install "dist/docsaidKit-${version}-none-any.whl[torch]"
   ```

## Frequently Asked Questions

1. **Why is Windows not supported?**

   To avoid potential issues, we recommend avoiding Windows.

2. **I insist on using Windows, what should I do?**

   In that case, we recommend installing Docker and using the above method to run your program within a Docker container.

   Please refer to the next section: [**Advanced Installation**](./advance.md).

3. **How do I install Docker?**

   It's not difficult, but it involves several steps.

   Refer to the [**Docker official documentation**](https://docs.docker.com/get-docker/) for installation instructions.
