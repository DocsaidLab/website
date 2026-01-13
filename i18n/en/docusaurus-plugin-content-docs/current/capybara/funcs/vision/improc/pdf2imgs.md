# pdf2imgs

> [pdf2imgs(stream: str | Path | bytes) -> list[np.ndarray] | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **Description**: Converts a PDF to a list of BGR numpy images (one per page).

- **Dependencies**

  - Requires system `poppler` (e.g. Ubuntu `poppler-utils`), otherwise `pdf2image` may not work properly.

- **Parameters**

  - **stream** (`str | Path | bytes`): PDF path or PDF bytes.

- **Returns**

  - **list[np.ndarray] | None**: Images per page; returns `None` on failure.

- **Example**

  ```python
  from capybara.vision.improc import pdf2imgs

  imgs = pdf2imgs('sample.pdf')
  ```
