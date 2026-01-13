# download_from_google

> [download_from_google(file_id: str, file_name: str, target: str | Path = ".") -> Path](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/utils.py)

- **Description**: Downloads a file from Google Drive and handles confirmation tokens for large files.

- **Parameters**

  - **file_id** (`str`): Google Drive file id.
  - **file_name** (`str`): File name to save as.
  - **target** (`str | Path`, optional): Target directory. Default is `"."`.

- **Returns**

  - **Path**: Path to the downloaded file.

- **Exceptions**

  - **Exception**: Failed to parse download link / confirmation parameters from the response.
  - **RuntimeError**: Failed during the file writing process.

- **Notes**

  - This function supports both small and large files. For large files, it handles Google's confirmation token automatically.

- **Example**

  ```python
  from capybara.utils import download_from_google

  path = download_from_google(
      file_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      file_name="example_file.txt",
  )

  path = download_from_google(
      file_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      file_name="example_file.txt",
      target="./downloads",
  )
  ```
