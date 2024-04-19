---
sidebar_position: 2
---

# Installation

Before installing DocsaidKit, please ensure your system meets the following requirements:

## Prerequisites

### Python Version

- Ensure that Python 3.8 or higher is installed on your system.

### Dependency Packages

Install the necessary dependencies based on your operating system.

- **Ubuntu**

    Open a terminal and execute the following commands to install dependencies:

    ```bash
    sudo apt install libturbojpeg exiftool ffmpeg libheif-dev
    ```

- **MacOS**

    Use brew to install dependencies:

    ```bash
    brew install jpeg-turbo exiftool ffmpeg libheif
    ```

### pdf2image Dependencies

pdf2image is a Python module used to convert PDF documents into images.

Follow the installation instructions based on your operating system:

- Refer to the open-source project [**pdf2image**](https://github.com/Belval/pdf2image) for installation guides.

- **MacOS**: Mac users need to install poppler via Brew:

    ```bash
    brew install poppler
    ```

- **Linux**: Most Linux distributions come pre-installed with `pdftoppm` and `pdftocairo`.

    If not installed, install poppler-utils via your package manager:

    ```bash
    sudo apt install poppler-utils
    ```

## Package Installation

Once the prerequisites are met, choose one of the following methods to install:

### Installation via git clone

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

### Installation via docker (recommended)

I always install via docker to ensure consistency of the environment, no exceptions.

I also recommend you install via docker. I have tested the environment, just use the following command:

```bash
cd DocsaidKit
bash docker/build.bash
```

After completion, whenever you want to use it, wrap your commands in docker:

```bash
docker run -v ${PWD}:/code -it docsaid_training_base_image your_scripts.py
```

For specific contents of the build file, refer to: [**Dockerfile**](https://github.com/DocsaidLab/DocsaidKit/blob/main/docker/Dockerfile)

## FAQs

1. **Why no Windows support?**

    Sorry, I have PTSD (Post-Traumatic Stress Disorder) from building environments on Windows, so it's not supported.

    Value your life, stay away from Windows.

2. **I want to use Windows anyway.**

    Alright, I recommend installing Docker and then using the Docker Image to run your programs.

3. **How do I install Docker?**

    It's not hard, but there are several steps.

    Refer to [**Docker's official documentation**](https://docs.docker.com/get-docker/) for installation instructions.