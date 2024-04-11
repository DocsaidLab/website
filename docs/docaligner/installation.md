---
sidebar_position: 2
---

# 安裝

目前沒有提供 Pypi 上的安裝包，短時間內也沒有相關規劃。

若要使用本專案，你必須直接從 Github 上 clone 本專案，然後安裝相依套件。

:::tip
安裝前請確認您已經安裝了 `DocsaidKit`。

如果你還沒有安裝 `DocsaidKit`，請參考 [**DocsaidKit 安裝指南**](../docsaidkit/installation)。
:::

## 安裝步驟

1. **Clone 本專案：**

    ```bash
    git clone https://github.com/DocsaidLab/DocAligner.git
    ```

2. **進入專案目錄：**

    ```bash
    cd DocAligner
    ```

3. **安裝相依套件：**

    ```bash
    pip install setuptools wheel
    ```

3. **建立打包文件：**

    ```bash
    python setup.py bdist_wheel
    ```

4. **安裝打包文件：**

    ```bash
    pip install dist/docaligner-*-py3-none-any.whl
    ```

遵循這些步驟，你應該能夠順利完成 `DocAligner` 的安裝。

安裝完成後即可以使用本專案。

## 測試安裝

你可以使用以下指令來測試安裝是否成功：

```bash
python -c "import docaligner; print(docaligner.__version__)"
# >>> 0.5.0
```

如果你看到類似 `0.5.0` 的版本號，則表示安裝成功。
