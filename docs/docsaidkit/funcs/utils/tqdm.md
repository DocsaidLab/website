---
sidebar_position: 6
---

# Tqdm

>[Tqdm(iterable=None, desc=None, smoothing=0, **kwargs)](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/custom_tqdm.py#L8)

- **說明**：這是一個繼承自 `tqdm` 的自定義進度條，用於迭代器的迭代過程中顯示進度條。我們在這裡對於原本 `tqdm` 的改動是在 `total` 參數上，當使用者沒有指定 `total` 時，我們會自動計算 `iterable` 的長度，並將其設定為 `total`。這樣的設計是為了讓使用者在不需要指定 `total` 的情況下，也能夠正確顯示進度條。

- **參數**
    - **iterable** (`Iterable`)：要迭代的對象。
    - **desc** (`str`)：進度條的描述。
    - **smoothing** (`int`)：平滑參數。
    - **kwargs** (`Any`)：其他參數。

- **範例**

    ```python
    import docsaidkit as D

    for i in D.Tqdm(range(100), desc='Processing'):
        pass
    ```

