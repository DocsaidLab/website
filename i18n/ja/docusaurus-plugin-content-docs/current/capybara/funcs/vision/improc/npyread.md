---
sidebar_position: 9
---

# npyread

> [npyread(path: Union[str, Path]) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L174)

- **説明**：NumPy の`.npy`ファイルから画像配列を読み込みます。

- 引数

  - **path** (`Union[str, Path]`)：`.npy`ファイルのパス。

- **返り値**

  - **np.ndarray**：読み込んだ画像配列。読み込みに失敗した場合は`None`を返します。

- **例**

  ```python
  import docsaidkit as D

  img = D.npyread('lena.npy')
  ```
