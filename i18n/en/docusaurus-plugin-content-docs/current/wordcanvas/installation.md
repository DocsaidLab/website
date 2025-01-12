---
sidebar_position: 2
---

# Installation

We offer installation via PyPI or by cloning this project from GitHub.

## Install via PyPI

1. Install `wordcanvas-docsaid`:

   ```bash
   pip install wordcanvas-docsaid
   ```

2. Verify the installation:

   ```bash
   python -c "import wordcanvas; print(wordcanvas.__version__)"
   ```

3. If you see the version number, the installation was successful.

## Install via GitHub

:::tip
To install via GitHub, make sure you have `Capybara` installed.

If not, refer to the [**Capybara Installation Guide**](../capybara/installation.md).
:::

1. **Clone the project:**

   ```bash
   git clone https://github.com/DocsaidLab/WordCanvas.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd WordCanvas
   ```

3. **Install the dependencies:**

   ```bash
   pip install setuptools wheel
   ```

4. **Build the package:**

   ```bash
   python setup.py bdist_wheel
   ```

5. **Install the package:**

   ```bash
   pip install dist/wordcanvas_docsaid-*-py3-none-any.whl
   ```

By following these steps, you should be able to successfully install `WordCanvas`.

Once the installation is complete, you can use the following command to test if the installation was successful:

```bash
python -c "import wordcanvas; print(wordcanvas.__version__)"
# >>> 2.0.0
```

If you see a version number like `2.0.0`, the installation was successful.
