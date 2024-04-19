---
sidebar_position: 3
---

# pdf2imgs

>[pdf2imgs(stream: Union[str, Path, bytes]) -> Union[List[np.ndarray], None]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/improc.py#L275C1-L292C15)

- **Description**: Convert a PDF file to a list of images in numpy format.

- **Parameters**
    - **stream** (`Union[str, Path, bytes]`): The path to the PDF file or binary data of the PDF.

- **Returns**
    - **List[np.ndarray]**: Returns a list of numpy images for each page of the PDF file if successful, otherwise returns None.

- **Example**

    ```python
    import docsaidkit as D

    imgs = D.pdf2imgs('sample.pdf')
    ```
