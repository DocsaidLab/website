# PowerDict

> [PowerDict(d=None, \*\*kwargs)](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/powerdict.py#L10)

- **Description**: This class is used to create a dictionary with freeze and melt functionality, allowing access to internal attributes via the `.` operator.

- **Attributes**

  - **is_frozen** (`bool`): Indicates whether the dictionary is frozen.

- **Methods**

  - **freeze()**: Freezes the dictionary.
  - **melt()**: Unfreezes the dictionary.
  - **to_dict()**: Converts the dictionary to a standard dictionary.
  - **to_json(path: Union[str, Path]) -> None**: Writes the dictionary to a JSON file.
  - **to_yaml(path: Union[str, Path]) -> None**: Writes the dictionary to a YAML file.
  - **to_txt(path: Union[str, Path]) -> None**: Writes the dictionary to a TXT file.
  - **to_pickle(path: Union[str, Path]) -> None**: Writes the dictionary to a pickle file.

- **Class Methods**

  - **load_json(path: Union[str, Path]) -> PowerDict**: Loads a dictionary from a JSON file.
  - **load_pickle(path: Union[str, Path]) -> PowerDict**: Loads a dictionary from a pickle file.
  - **load_yaml(path: Union[str, Path]) -> PowerDict**: Loads a dictionary from a YAML file.

- **Parameters**

  - **d** (`dict`): The dictionary to be converted. Default is None.

- **Example**

  ```python
  from capybara import PowerDict

  d = {'key': 'value'}
  cfg = PowerDict(d)
  print(cfg.key)
  # >>> 'value'
  ```
