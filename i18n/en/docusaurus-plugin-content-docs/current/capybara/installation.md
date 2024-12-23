---
sidebar_position: 2
---

# Installation

Before installing Capybara, make sure your system meets the following requirements:

## Prerequisites

Ensure that Python 3.10 or higher is installed on your system before proceeding with the installation.

We developed this based on the Ubuntu operating system, so the following instructions may not apply to Windows and MacOS users.

Next, open your terminal and run the following command to install dependencies:

```bash
sudo apt install libturbojpeg exiftool ffmpeg libheif-dev
```

### pdf2image Dependencies

pdf2image is a Python module used to convert PDF files into images.

Follow the instructions below to install it based on your operating system:

- Alternatively, refer to the open-source project [**pdf2image**](https://github.com/Belval/pdf2image) for installation guides.

Most Linux distributions come pre-installed with `pdftoppm` and `pdftocairo`.

If not installed, use your package manager to install poppler-utils:

```bash
sudo apt install poppler-utils
```

## Installing the Package

Once the prerequisites are met, you can install the package via git clone:

1. Clone the repository:

   ```bash
   git clone https://github.com/DocsaidLab/Capybara.git
   ```

2. Install the wheel package:

   ```bash
   pip install wheel
   ```

3. Build the wheel file:

   ```bash
   cd Capybara
   python setup.py bdist_wheel
   ```

4. Install the built wheel package:

   ```bash
   pip install dist/capybara-*-py3-none-any.whl
   ```

## Frequently Asked Questions

1. **Why is there no Windows version?**

   For the love of life, stay away from Windows.

2. **I just want to use Windows, and I prefer you not to meddle!**

   Fine, we suggest installing Docker and then using the methods above to run your program via Docker.

   Please refer to the next section: [**Advanced Installation**](./advance.md).

3. **How do I install Docker?**

   It's not difficult, but there are a few steps.

   Please refer to the [**Docker Official Documentation**](https://docs.docker.com/get-docker/) for installation instructions.
