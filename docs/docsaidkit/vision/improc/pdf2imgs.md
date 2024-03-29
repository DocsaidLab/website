---
sidebar_position: 3
---

# pdf2imgs


```python
def pdf2imgs(stream: Union[str, Path, bytes]) -> Union[List[np.ndarray], None]:
    """
    Function for converting a PDF document to numpy images.

    Args:
        file_dir (str): A path of PDF document.

    Returns:
        img: Images will be a list of np image representing each page of the PDF document.
    """
    try:
        if isinstance(stream, bytes):
            pil_imgs = convert_from_bytes(stream)
        else:
            pil_imgs = convert_from_path(stream)
        return [imcvtcolor(np.array(img), cvt_mode='RGB2BGR') for img in pil_imgs]
    except:
        return
```

>[pdf2imgs(stream: Union[str, Path, bytes]) -> Union[List[np.ndarray], None]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/improc.py#L275C1-L292C15)

- **說明**：將 PDF 文件轉換為 numpy 格式的圖片列表。

- **參數**

    - **stream** (`Union[str, Path, bytes]`)：PDF 文件的路徑或二進制數據。

- **傳回值**

    - **List[np.ndarray]**：成功時返回 PDF 文件的每一頁的 numpy 圖片列表，否則返回 None。

- **範例**

    ```python
    import docsaidkit as D

    imgs = D.pdf2imgs('sample.pdf')
    ```
