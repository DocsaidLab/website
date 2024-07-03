---
sidebar_position: 3
---

# 安裝套件

目前沒有提供 Pypi 上的安裝包，短時間內也沒有相關規劃。

若要使用本專案，你必須直接從 Github 上 clone 本專案，然後安裝相依套件。

## 安裝步驟

1. **Clone 本專案：**

   ```bash
   git clone https://github.com/DocsaidLab/AutoTraderX.git
   ```

2. **進入專案目錄：**

   ```bash
   cd AutoTraderX
   ```

3. **安裝相依套件：**

   ```bash
   pip install setuptools wheel
   ```

4. **建立打包文件：**

   ```bash
   python setup.py bdist_wheel
   ```

5. **安裝打包文件：**

   ```powershell
   pip install dist\autotraderx-*-py3-none-any.whl
   ```

遵循這些步驟，你應該能夠順利完成 `AutoTraderX` 的安裝。

### 安裝元富證券 Python API

從元富證券官方網站下載 Python API：

- [**元富證券-下載專區**](https://mlapi.masterlink.com.tw/web_api/service/home#download)

  ![download](./img/download.jpg)

下載後解壓縮，並使用 pip 安裝：

```powershell
pip install .\MasterTradePy\MasterTradePy\64bit\MasterTradePy-0.0.23-py3-none-win_amd64.whl
pip install .\Python_tech_analysis\tech_analysis_api_v2-0.0.5-py3-none-win_amd64.whl
pip install .\SolPYAPI\PY_TradeD-0.1.15-py3-none-any.whl
```

安裝完成後即可以使用本專案。

:::tip
本專案亦有提供元富證券 Python API 的 .whl 安裝檔案，在 `MasterLink_PythonAPI` 資料夾中。

你可以直接執行以下指令安裝：

```powershell
.\run_install.bat
```

請注意我們不會更新這些檔案，請自行至元富證券官方網站下載最新版本。
:::

## 測試安裝

你可以使用以下指令來測試安裝是否成功：

```bash
python -c "import autotraderx; print(autotraderx.__version__)"
# >>> 0.1.0
```

如果你看到類似 `0.1.0` 的版本號，則表示安裝成功。
