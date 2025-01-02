---
sidebar_position: 2
---

# 安裝

我們有提供 PyPI 上的安裝，或是從 Github 上 clone 本專案的方式來安裝。

## 透過 PyPI 安裝

1. 安裝 `mrzscanner-docsaid`：

   ```bash
   pip install mrzscanner-docsaid
   ```

2. 驗證安裝：

   ```bash
   python -c "import `mrzscanner; print(`mrzscanner.__version__)"
   ```

3. 如果你看到版本號，則表示安裝成功。

## 透過 GitHub 安裝

:::tip
若要透過 GitHub 安裝，請確保你已經安裝了 `Capybara`。

如果沒有，請參考 [**Capybara 安裝指南**](../capybara/installation.md)。
:::

1. **Clone 本專案：**

   ```bash
   git clone https://github.com/DocsaidLab/MRZScanner.git
   ```

2. **進入專案目錄：**

   ```bash
   cd MRZScanner
   ```

3. **安裝相依套件：**

   ```bash
   pip install wheel
   ```

4. **建立打包文件：**

   ```bash
   python setup.py bdist_wheel
   ```

5. **安裝打包文件：**

   ```bash
   pip install dist/mrzscanner_docsaid-*-py3-none-any.whl
   ```

遵循這些步驟，你應該能夠順利完成 `MRZScanner` 的安裝。

安裝完成後，可以使用以下指令來測試安裝是否成功：

```bash
python -c "import mrzscanner; print(mrzscanner.__version__)"
# >>> 0.3.2
```

如果你看到類似 `0.3.2` 的版本號，則表示安裝成功。
