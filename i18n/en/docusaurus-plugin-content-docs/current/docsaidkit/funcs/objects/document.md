---
sidebar_position: 1
---

# Document

> [Document(image: Optional[np.ndarray] = None, doc_polygon: Optional[Polygon] = None, doc_type: Optional[str] = None, ocr_texts: Optional[List[str] = None, ocr_polygons: Optional[Polygons] = None, ocr_kie: Optional[dict] = None)]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/document.py#L61)

- **Description**

    Document class, which includes document image, document polygon, document type, OCR texts, OCR polygons, and OCR key information extraction (KIE). This class is used to integrate the outputs of multiple models and consolidate various aspects of information.

- **Parameters**
    - **image** (`Optional[np.ndarray]`): Document image.
    - **doc_polygon** (`Optional[Polygon]`): Document polygon.
    - **doc_type** (`Optional[str]`): Document type.
    - **ocr_texts** (`Optional[List[str]]`): OCR texts.
    - **ocr_polygons** (`Optional[Polygons]`): OCR polygons.
    - **ocr_kie** (`Optional[dict]`): OCR key information extraction.

- **Attributes**
    - **has_doc_polygon**: Indicates whether there is a document polygon.
    - **has_ocr_polygons**: Indicates whether there are OCR polygons.
    - **has_ocr_texts**: Indicates whether there are OCR texts.
    - **doc_flat_img**: Cropped image based on the document polygon.
    - **doc_polygon_angle**: Angle of the document polygon.

- **Methods**
    - **be_jsonable(exclude_image: bool = True) -> dict**: Converts the document class to a JSON serializable dictionary.
    - **gen_doc_flat_img(image_size: Optional[Tuple[int, int]] = None) -> np.ndarray**: Generates the cropped image based on the document polygon.
    - **gen_doc_info_image(thickness: Optional[int] = None) -> np.ndarray**: Generates an information image of the document polygon.
    - **gen_ocr_info_image() -> np.ndarray**: Generates an information image of the OCR.
    - **get_path(folder: str = None, name: str = None) -> Path**: Gets the path of the document.
    - **draw_doc(folder: Optional[str] = None, name: Optional[str] = None, **kwargs) -> None**: Draws an information image of the document polygon.
    - **draw_ocr(folder: Optional[str] = None, name: Optional[str] = None, **kwargs) -> None**: Draws an information image of the OCR.

- **Example**

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
