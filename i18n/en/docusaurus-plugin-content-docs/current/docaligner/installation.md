---
sidebar_position: 2
---

# Installation

We provide installation via PyPI or by cloning the project from GitHub.

## Install via PyPI

1. Install `docaligner-docsaid`:

   ```bash
   pip install docaligner-docsaid
   ```

2. Verify the installation:

   ```bash
   python -c "import docaligner; print(docaligner.__version__)"
   ```

3. If you see the version number, the installation is successful.

## Install via GitHub

:::tip
If you want to install via GitHub, make sure you have installed `Capybara`.

If not, refer to the [**Capybara Installation Guide**](../capybara/installation.md).
:::

1. **Clone the project:**

   ```bash
   git clone https://github.com/DocsaidLab/DocAligner.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd DocAligner
   ```

3. **Install dependencies:**

   ```bash
   pip install setuptools wheel
   ```

4. **Create the package:**

   ```bash
   python setup.py bdist_wheel
   ```

5. **Install the package:**

   ```bash
   pip install dist/docaligner_docsaid-*-py3-none-any.whl
   ```

By following these steps, you should be able to successfully install `DocAligner`.

Once installed, You can test the installation by running the following command:

```bash
python -c "import docaligner; print(docaligner.__version__)"
# >>> 1.0.0
```

If you see a version number like `1.1.0`, the installation was successful.
