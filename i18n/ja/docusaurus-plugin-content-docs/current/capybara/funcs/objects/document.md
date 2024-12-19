---
sidebar_position: 1
---

# Document

> [Document(image: Optional[np.ndarray] = None, doc_polygon: Optional[Polygon] = None, doc_type: Optional[str] = None, ocr_texts: Optional[List[str]] = None, ocr_polygons: Optional[Polygons] = None, ocr_kie: Optional[dict] = None)](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/document.py#L61)

- **説明**：ドキュメントクラスで、ドキュメント画像、ドキュメント多角形、ドキュメントタイプ、OCR テキスト、OCR 多角形、OCR KIE などを含みます。このクラスは、複数のモデルからの出力を統合し、さまざまな情報を整理します。

- **パラメータ**

  - **image** (`Optional[np.ndarray]`)：ドキュメント画像。
  - **doc_polygon** (`Optional[Polygon]`)：ドキュメント多角形。
  - **doc_type** (`Optional[str]`)：ドキュメントタイプ。
  - **ocr_texts** (`Optional[List[str]]`)：OCR テキスト。
  - **ocr_polygons** (`Optional[Polygons]`)：OCR 多角形。
  - **ocr_kie** (`Optional[dict]`)：OCR KIE。

- **属性**

  - **has_doc_polygon**：ドキュメント多角形があるかどうか。
  - **has_ocr_polygons**：OCR 多角形があるかどうか。
  - **has_ocr_texts**：OCR テキストがあるかどうか。
  - **doc_flat_img**：ドキュメント多角形で切り取られた画像。
  - **doc_polygon_angle**：ドキュメント多角形の角度。

- **メソッド**

  - **be_jsonable(exclude_image: bool = True) -> dict**：ドキュメントクラスを JSON シリアライズ可能な辞書に変換。
  - **gen_doc_flat_img(image_size: Optional[Tuple[int, int]] = None) -> np.ndarray**：ドキュメント多角形で切り取られた画像を生成。
  - **gen_doc_info_image(thickness: Optional[int] = None) -> np.ndarray**：ドキュメント多角形の情報画像を生成。
  - **gen_ocr_info_image() -> np.ndarray**：OCR の情報画像を生成。
  - **get_path(folder: str = None, name: str = None) -> Path**：ドキュメントのパスを取得。
  - **draw_doc(folder: Optional[str] = None, name: Optional[str] = None, **kwargs) -> None\*\*：ドキュメント多角形の情報画像を描画。
  - **draw_ocr(folder: Optional[str] = None, name: Optional[str] = None, **kwargs) -> None\*\*：OCR の情報画像を描画。

- **例**

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
