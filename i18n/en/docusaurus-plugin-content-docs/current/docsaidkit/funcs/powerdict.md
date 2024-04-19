---
sidebar_position: 5
---

# PowerDict

>[PowerDict(d=None, **kwargs)](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/powerdict.py#L10)

- **Description**: This class is designed to create a dictionary with freezing and unfreezing capabilities, and allows accessing its properties through dot notation.

- **Attributes**

    - **is_frozen** (`bool`): Determines whether the dictionary is frozen.

- **Methods**

    - **freeze()**: Freezes the dictionary.
    - **melt()**: Unfreezes the dictionary.
    - **to_dict()**: Converts the dictionary to a standard dictionary.
    - **to_json(path: Union[str, Path]) -> None**: Writes the dictionary to a JSON file.
    - **to_yaml(path: Union[str, Path]) -> None**: Writes the dictionary to a YAML file.
    - **to_txt(path: Union[str, Path]) -> None**: Writes the dictionary to a text file.
    - **to_pickle(path: Union[str, Path]) -> None**: Writes the dictionary to a pickle file.

- **Class Methods**

    - **load_json(path: Union[str, Path]) -> PowerDict**: Loads a dictionary from a JSON file.
    - **load_pickle(path: Union[str, Path]) -> PowerDict**: Loads a dictionary from a pickle file.
    - **load_yaml(path: Union[str, Path]) -> PowerDict**: Loads a dictionary from a YAML file.

- **Parameters**
    - **d** (`dict`): The dictionary to convert. Default is None.

- **Example**

    ```python
    from docsaidkit import PowerDict

    d = {'key': 'value'}
    cfg = PowerDict(d)
    print(cfg.key)
    # >>> 'value'
    ```