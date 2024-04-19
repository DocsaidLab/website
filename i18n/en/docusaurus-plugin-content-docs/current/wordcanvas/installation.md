---
sidebar_position: 2
---

# Installation

Currently, there is no package available on PyPI, and there are no plans to provide one in the near future. To use this project, you must clone it directly from Github and then install the required dependencies.

:::tip
Before installation, please ensure you have installed `DocsaidKit`.

If you have not installed `DocsaidKit`, please refer to the [**DocsaidKit Installation Guide**](../docsaidkit/installation).
:::

## Installation Steps

1. **Clone the project:**

    ```bash
    git clone https://github.com/DocsaidLab/WordCanvas.git
    ```

2. **Enter the project directory:**

    ```bash
    cd WordCanvas
    ```

3. **Install dependencies:**

    ```bash
    pip install wheel
    ```

4. **Build the package:**

    ```bash
    python setup.py bdist_wheel
    ```

5. **Install the package:**

    ```bash
    pip install dist/wordcanvas-*-py3-none-any.whl
    ```

Following these steps, you should be able to successfully install `WordCanvas`.

Once installed, you are ready to use the project.

## Test the Installation

You can test whether the installation was successful with the following command:

```bash
python -c "import wordcanvas; print(wordcanvas.__version__)"
# >>> 0.1.0
```

If you see a version number similar to `0.1.0`, it indicates the installation was successful.