---
sidebar_position: 7
---

# draw_ocr_infos

> [draw_ocr_infos(img: np.ndarray, texts: List[str], polygons: Polygons, colors: tuple = None, concat_axis: int = 1, thicknesses: int = 2, font_path: str = None) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/visualization/draw.py#L308)

- **說明**：在影像上繪製 OCR 結果。

- **參數**
    - **img** (`np.ndarray`)：要繪製的影像。
    - **texts** (`List[str]`)：偵測到的文字字串列表。
    - **polygons** (`D.Polygons`)：表示偵測到的文字邊界的多邊形列表。
    - **colors** (`tuple`)：繪製顏色的 RGB 值。如果未提供，則為每個文字根據固定的邏輯生成唯一顏色。
    - **concat_axis** (`int`)：用於串接原始影像和標註影像的軸。預設為 1 (水平)。
    - **thicknesses** (`int`)：繪製多邊形的粗細。預設為 2。
    - **font_path** (`str`)：要使用的字型檔案的路徑。如果未提供，則使用預設字型 "NotoSansMonoCJKtc-VF.ttf"。

- **傳回值**
    - **np.ndarray**：串接了原始影像和標註影像的影像。

- **範例**

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    texts = ['Hello!', 'Test!']
    polygons = [
        D.Polygon([(20, 20), (100, 20), (80, 80), (20, 40)]),
        D.Polygon([(100, 200), (20, 200), (40, 140), (100, 180)])
    ]
    ocr_img = D.draw_ocr_infos(img, texts, polygons, concat_axis=1, thicknesses=2)
    ```

    ![draw_ocr_infos](./resource/test_draw_ocr_infos.jpg)
