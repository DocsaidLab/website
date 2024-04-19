---
sidebar_position: 6
---

# draw_mask

> [draw_mask(img: np.ndarray, mask: np.ndarray, colormap: int = cv2.COLORMAP_JET, weight: Tuple[float, float] = (0.5, 0.5), gamma: float = 0, min_max_normalize: bool = False) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/visualization/draw.py#L366)

- **Description**

    Draw a mask on an image.

- **Parameters**

    - **img** (`np.ndarray`): The image to draw on.
    - **mask** (`np.ndarray`): The mask to draw.
    - **colormap** (`int`): The colormap used for the mask. Defaults to `cv2.COLORMAP_JET`.
    - **weight** (`Tuple[float, float]`): The weights of the image and the mask. Defaults to (0.5, 0.5).
    - **gamma** (`float`): The gamma value of the mask. Defaults to 0.
    - **min_max_normalize** (`bool`): Whether to normalize the mask to the range [0, 1]. Defaults to False.

- **Returns**

    - **np.ndarray**: The image with the drawn mask.

- **Example**

    ```python
    import docsaidkit as D
    import numpy as np

    img = D.imread('lena.png')
    polygon = D.Polygon([(20, 20), (100, 20), (80, 80), (20, 40)])
    mask = D.draw_polygon(np.zeros_like(img), polygon, fillup=True, color=255)
    mask_img = D.draw_mask(img, mask, colormap=cv2.COLORMAP_JET, weight=(0.5, 0.5), gamma=0, min_max_normalize=False)
    ```

    ![draw_mask](./resource/test_draw_mask.jpg)
