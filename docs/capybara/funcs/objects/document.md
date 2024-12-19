---
sidebar_position: 1
---

# Document

> [Document(image: Optional[np.ndarray] = None, doc_polygon: Optional[Polygon] = None, doc_type: Optional[str] = None, ocr_texts: Optional[List[str] = None, ocr_polygons: Optional[Polygons] = None, ocr_kie: Optional[dict] = None)]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/document.py#L61)

- **說明**：文件類別，包含文件影像、文件多邊形、文件類型、OCR 文字、OCR 多邊形、OCR KIE。這個類別用來整合多個模型的輸出，彙整各方面的資訊。

- **參數**
    - **image** (`Optional[np.ndarray]`)：文件影像。
    - **doc_polygon** (`Optional[Polygon]`)：文件多邊形。
    - **doc_type** (`Optional[str]`)：文件類型。
    - **ocr_texts** (`Optional[List[str]]`)：OCR 文字。
    - **ocr_polygons** (`Optional[Polygons]`)：OCR 多邊形。
    - **ocr_kie** (`Optional[dict]`)：OCR KIE。

- **屬性**
    - **has_doc_polygon**：是否有文件多邊形。
    - **has_ocr_polygons**：是否有 OCR 多邊形。
    - **has_ocr_texts**：是否有 OCR 文字。
    - **doc_flat_img**：文件多邊形裁剪後的影像。
    - **doc_polygon_angle**：文件多邊形的角度。

- **方法**
    - **be_jsonable(exclude_image: bool = True) -> dict**：將文件類別轉換為 JSON 可序列化的字典。
    - **gen_doc_flat_img(image_size: Optional[Tuple[int, int]] = None) -> np.ndarray**：生成文件多邊形裁剪後的影像。
    - **gen_doc_info_image(thickness: Optional[int] = None) -> np.ndarray**：生成文件多邊形的信息影像。
    - **gen_ocr_info_image() -> np.ndarray**：生成 OCR 的信息影像。
    - **get_path(folder: str = None, name: str = None) -> Path**：取得文件的路徑。
    - **draw_doc(folder: Optional[str] = None, name: Optional[str] = None, **kwargs) -> None**：繪製文件多邊形的信息影像。
    - **draw_ocr(folder: Optional[str] = None, name: Optional[str] = None, **kwargs) -> None**：繪製 OCR 的信息影像。

- **範例**

    ```python
    import docsaidkit as D

    doc = D.Document()
    doc.doc_polygon = D.Polygon([[100, 100], [200, 100], [200, 200], [100, 200]])
    doc.doc_type = 'ID'
    doc.ocr_texts = ['Name: Alice', 'ID: 123456']
    doc.ocr_polygons = D.Polygons([[[100, 100], [200, 100], [200, 200], [100, 200]]])
    doc.ocr_kie = {'Name': 'Alice', 'ID': '123456'}
    print(doc)
    # >>> Document(
    #       image=None,
    #       doc_polygon=Polygon([[100, 100],
    #                            [200, 100],
    #                            [200, 200],
    #                            [100, 200]]),
    #       doc_type='ID',
    #       ocr_texts=['Name: Alice', 'ID: 123456'],
    #       ocr_polygons=Polygons([[[100, 100],
    #                               [200, 100],
    #                               [200, 200],
    #                               [100, 200]]]),
    #       ocr_kie={'Name': 'Alice', 'ID': '123456'})
    ```
