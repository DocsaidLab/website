---
sidebar_position: 2
---

# Installation

Currently, there is no installation package available on PyPI, and there are no immediate plans to release one.

To use this project, you must clone it directly from GitHub and install the required dependencies.

:::tip
Before installing, make sure you have already installed `DocsaidKit`.

If you haven't installed `DocsaidKit` yet, please refer to the [**DocsaidKit Installation Guide**](../docsaidkit/installation).
:::

## Installation Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/DocsaidLab/MRZScanner.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd MRZScanner
   ```

3. **Install required dependencies:**

   ```bash
   pip install wheel
   ```

4. **Build the package:**

   ```bash
   python setup.py bdist_wheel
   ```

5. **Install the built package:**

   ```bash
   pip install dist/mrzscanner-*-py3-none-any.whl
   ```

By following these steps, you should successfully complete the installation of `MRZScanner`.

Once the installation is finished, you can start using the project.

## Verify Installation

You can test if the installation was successful by running the following command:

```bash
python -c "import mrzscanner; print(mrzscanner.__version__)"
# >>> 0.1.0
```

If you see a version number like `0.1.0`, the installation was successful.
