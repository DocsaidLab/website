# Tqdm

> [Tqdm(iterable=None, desc=None, smoothing=0, **kwargs)](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/custom_tqdm.py)

- **Description**: This is a custom progress bar derived from `tqdm` to display a progress bar during the iteration process of an iterable. The modification made to the original `tqdm` is with the `total` parameter. When the user does not specify `total`, the length of the `iterable` is automatically calculated and set as `total`. This design ensures that the progress bar is correctly displayed without requiring the user to manually set `total`.

- **Parameters**

  - **iterable** (`Iterable`): The object to iterate over.
  - **desc** (`str`): A description for the progress bar.
  - **smoothing** (`int`): The smoothing parameter.
  - **kwargs** (`Any`): Other parameters.

- **Example**

  ```python
  from capybara.utils.custom_tqdm import Tqdm

  for i in Tqdm(range(100), desc='Processing'):
      pass
  ```
