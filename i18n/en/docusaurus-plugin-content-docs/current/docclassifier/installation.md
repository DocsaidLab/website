---
sidebar_position: 2
---

# Installation

Currently, there is no installation package available on PyPI, and there are no plans for one in the near future.

To use this project, you must clone the repository directly from GitHub and then install the required dependencies.

:::tip
Before installing, please ensure that you have installed `DocsaidKit`.

If you haven't installed `DocsaidKit` yet, please refer to the [**DocsaidKit Installation Guide**](../docsaidkit/installation).
:::

## Installation Steps

1. **Clone the repository:**

    ```bash
    git clone https://github.com/DocsaidLab/DocClassifier.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd DocClassifier
    ```

3. **Install the required dependencies:**

    ```bash
    pip install setuptools wheel
    ```

4. **Create the distribution package:**

    ```bash
    python setup.py bdist_wheel
    ```

5. **Install the distribution package:**

    ```bash
    pip install dist/docclassifier-*-py3-none-any.whl
    ```

By following these steps, you should be able to successfully install `DocClassifier`.

## Testing the Installation

You can test if the installation was successful by using the following command:

```bash
python -c "import docclassifier; print(docclassifier.__version__)"
# >>> 0.8.0
```

If you see a version number similar to `0.8.0`, it means the installation was successful.