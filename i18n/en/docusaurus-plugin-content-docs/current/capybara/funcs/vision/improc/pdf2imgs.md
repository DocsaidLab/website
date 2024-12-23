# pdf2imgs

> [pdf2imgs(stream: Union[str, Path, bytes]) -> Union[List[np.ndarray], None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L275)

- **Description**: Converts a PDF file into a list of images in numpy format.

- **Parameters**:

  - **stream** (`Union[str, Path, bytes]`): The path or binary data of the PDF file.

- **Return value**:

  - **List[np.ndarray]**: Returns a list of numpy images for each page of the PDF if successful; otherwise, returns None.

- **Example**:

  ```python
  import capybara as cb

  imgs = cb.pdf2imgs('sample.pdf')
  ```
