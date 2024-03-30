---
sidebar_position: 1
---

# Document


```python
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .mixins import DataclassCopyMixin, DataclassToJsonMixin
from .structures import Polygon, Polygons
from .utils import now
from .vision import draw_ocr_infos, imwarp_quadrangle, imwrite

__all__ = ['Document', 'poly_angle', 'calc_angle']


def calc_angle(v1, v2):
    """
    Calculate the angle between two vectors.
    """
    # Ensure the dot product is within the valid range for arccos
    dot_product = np.dot(v1, v2)
    norms_product = np.linalg.norm(v1, 2) * np.linalg.norm(v2, 2)
    cos_angle = np.clip(dot_product / norms_product, -1.0, 1.0)

    angle = np.arccos(cos_angle)
    angle = np.degrees(angle)

    # Determine the direction of the angle
    v1_3d = np.array([*v1, 0])
    v2_3d = np.array([*v2, 0])
    if np.cross(v1_3d, v2_3d)[-1] < 0:
        angle = 360 - angle

    return angle


def poly_angle(
    poly1: Polygon,
    poly2: Optional[Polygon] = None,
    base_vector: Tuple[int, int] = (0, 1)
) -> float:
    """
    Calculate the angle between two polygons or a polygon and a base vector.
    """

    def _get_angle(poly):
        poly_points = poly.numpy()
        vector1 = poly_points[2] - poly_points[0]
        vector2 = poly_points[3] - poly_points[1]
        return vector1 + vector2

    v1 = _get_angle(poly1)
    v2 = _get_angle(poly2) if poly2 is not None else np.array(
        base_vector, dtype='float32')

    return calc_angle(v1, v2)


@dataclass
class Document(DataclassCopyMixin, DataclassToJsonMixin):

    image: Optional[np.ndarray] = field(default=None)
    doc_polygon: Optional[Polygon] = field(default=None)
    doc_type: Optional[str] = field(default=None)
    ocr_texts: Optional[List[str]] = field(default=None)
    ocr_polygons: Optional[Polygons] = field(default=None)
    ocr_kie: Optional[dict] = field(default=None)

    @property
    def has_doc_polygon(self):
        return self.doc_polygon is not None

    @property
    def has_ocr_polygons(self):
        return self.ocr_polygons is not None

    @property
    def has_ocr_texts(self):
        return self.ocr_texts is not None

    def be_jsonable(self, exclude_image: bool = True) -> dict:
        if exclude_image and 'image' in self.__dict__:
            img = self.__dict__.pop('image')
            out = super().be_jsonable()
            self.image = img
            return out
        return super().be_jsonable()

    @property
    def doc_flat_img(self):
        return self.gen_doc_flat_img()

    @property
    def doc_polygon_angle(self):
        return poly_angle(self.doc_polygon)

    def gen_doc_flat_img(self, image_size: Optional[Tuple[int, int]] = None):
        if not self.has_doc_polygon:
            warnings.warn(
                'No polygon in the image, returns the original image.')
            return self.image.copy()

        if image_size is None:
            return imwarp_quadrangle(self.image, self.doc_polygon)

        h, w = image_size
        point1 = self.doc_polygon.astype('float32')
        point2 = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype='float32')
        M = cv2.getPerspectiveTransform(point1, point2)
        flat_img = cv2.warpPerspective(self.image, M, (int(w), int(h)))
        return flat_img

    def gen_doc_info_image(self, thickness: Optional[int] = None) -> np.ndarray:
        if not self.has_doc_polygon:
            warnings.warn(
                'No polygon in the image, returns the original image.')
            return self.image.copy()

        colors = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (0, 0, 255)]
        export_img = self.image.copy()
        _polys = self.doc_polygon.astype(int)
        _polys_roll = np.roll(_polys, 1, axis=0)
        for p1, p2, color in zip(_polys, _polys_roll, colors):
            _thickness = max(int(export_img.shape[1] * 0.005), 1) \
                if thickness is None else thickness
            export_img = cv2.circle(
                export_img, p2, radius=_thickness*2,
                color=color, thickness=-1, lineType=cv2.LINE_AA
            )
            export_img = cv2.arrowedLine(
                export_img, p2, p1, color=color,
                thickness=_thickness, line_type=cv2.LINE_AA
            )
        return export_img

    def gen_ocr_info_image(self, **kwargs):
        if self.has_ocr_polygons and self.has_ocr_texts:
            return draw_ocr_infos(self.image, self.ocr_texts, self.ocr_polygons, **kwargs)
        return self.image.copy()

    def get_path(self, folder: str = None, name: str = None) -> Path:
        folder = Path(folder or '.')
        name = name or f'output_{now()}.jpg'
        return folder / name

    def draw_doc(self, folder: Optional[str] = None, name: Optional[str] = None, **kwargs) -> None:
        imwrite(self.gen_doc_info_image(**kwargs), self.get_path(folder, name))

    def draw_ocr(self, folder: Optional[str] = None, name: Optional[str] = None, **kwargs) -> None:
        imwrite(self.gen_ocr_info_image(**kwargs), self.get_path(folder, name))
```

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
