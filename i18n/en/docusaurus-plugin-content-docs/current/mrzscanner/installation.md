---
sidebar_position: 2
---

# Installation

We provide installation through PyPI or by cloning the project from GitHub.

## Install via PyPI

1. Install `mrzscanner-docsaid`:

   ```bash
   pip install mrzscanner-docsaid
   ```

2. Verify the installation:

   ```bash
   python -c "import mrzscanner; print(mrzscanner.__version__)"
   ```

3. If you see the version number, the installation is successful.

## Install via GitHub

:::tip
To install via GitHub, make sure you have `Capybara` installed.

If not, refer to the [**Capybara Installation Guide**](../capybara/installation.md).
:::

1. **Clone the project:**

   ```bash
   git clone https://github.com/DocsaidLab/MRZScanner.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd MRZScanner
   ```

3. **Install dependencies:**

   ```bash
   pip install wheel
   ```

4. **Create the package:**

   ```bash
   python setup.py bdist_wheel
   ```

5. **Install the package:**

   ```bash
   pip install dist/mrzscanner_docsaid-*-py3-none-any.whl
   ```

By following these steps, you should be able to successfully install `MRZScanner`.

Once installed, you can test the installation with the following command:

```bash
python -c "import mrzscanner; print(mrzscanner.__version__)"
# >>> 1.0.6
```

If you see a version number like `1.0.6`, the installation was successful.
