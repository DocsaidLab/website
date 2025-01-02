---
sidebar_position: 2
---

# Installation

We provide installation via PyPI or by cloning the project from GitHub.

## Installation via PyPI

1. Install `docclassifier-docsaid`:

   ```bash
   pip install docclassifier-docsaid
   ```

2. Verify the installation:

   ```bash
   python -c "import docclassifier; print(docclassifier.__version__)"
   ```

3. If you see the version number, the installation is successful.

## Installation via GitHub

:::tip
To install via GitHub, make sure you have `Capybara` installed.

If not, please refer to the [**Capybara Installation Guide**](../capybara/installation.md).
:::

1. **Clone the project:**

   ```bash
   git clone https://github.com/DocsaidLab/DocClassifier.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd DocClassifier
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
   pip install dist/docclassifier_docsaid-*-py3-none-any.whl
   ```

By following these steps, you should be able to successfully install `DocClassifier`.

After installation, you can use the following command to test if the installation was successful:

```bash
python -c "import docclassifier; print(docclassifier.__version__)"
# >>> 0.10.0
```

If you see a version number like `0.10.0`, the installation was successful.
