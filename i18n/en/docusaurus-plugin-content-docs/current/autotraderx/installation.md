---
sidebar_position: 3
---

# Installation

Currently, there is no installation package available on PyPI, and there are no immediate plans for it.

To use this project, you need to clone it directly from GitHub and install the dependencies.

## Installation Steps

1. **Clone the project:**

   ```bash
   git clone https://github.com/DocsaidLab/AutoTraderX.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd AutoTraderX
   ```

3. **Install dependencies:**

   ```bash
   pip install setuptools wheel
   ```

4. **Build the package:**

   ```bash
   python setup.py bdist_wheel
   ```

5. **Install the package:**

   ```powershell
   pip install dist\autotraderx-*-py3-none-any.whl
   ```

By following these steps, you should successfully install `AutoTraderX`.

### Installing MasterLink Python API

Download the Python API from the MasterLink official website:

- [**MasterLink - Download Area**](https://mlapi.masterlink.com.tw/web_api/service/home#download)

  ![download](./img/download.jpg)

After downloading, unzip the files and install them using pip:

```powershell
pip install .\MasterTradePy\MasterTradePy\64bit\MasterTradePy-0.0.23-py3-none-win_amd64.whl
pip install .\Python_tech_analysis\tech_analysis_api_v2-0.0.5-py3-none-win_amd64.whl
pip install .\SolPYAPI\PY_TradeD-0.1.15-py3-none-any.whl
```

Once installed, you can use this project.

:::tip
This project also provides .whl installation files for the MasterLink Python API in the `MasterLink_PythonAPI` folder.

You can directly run the following command for installation:

```powershell
.\run_install.bat
```

Please note that we do not update these files; please download the latest versions from the MasterLink official website.
:::

## Testing Installation

You can test if the installation was successful with the following command:

```bash
python -c "import autotraderx; print(autotraderx.__version__)"
# >>> 0.1.0
```

If you see a version number like `0.1.0`, the installation was successful.
