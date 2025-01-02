# download

## download_from_google

> [download_from_google(file_id: str, file_name: str, target: str = ".") -> None](https://github.com/DocsaidLab/Capybara/blob/c83d96363a3de3686a98dd7b558a168f08b9bc97/capybara/utils/utils.py#L70)

- **Description**: Download files from Google Drive and handle confirmation issues for large files.

- **Parameters**

  - **file_id** (`str`): The ID of the file to be downloaded from Google Drive.
  - **file_name** (`str`): The name to save the file as after downloading.
  - **target** (`str`, optional): The target directory to save the file. The default is the current directory `"."`.

- **Exceptions**

  - **Exception**: An exception is raised if the download fails or the file cannot be created.

- **Notes**

  - This function handles both small and large files. For large files, it automatically handles Google's confirmation checks, bypassing warnings about virus scans or file size limits.
  - Ensure the target directory exists; if not, it will be created automatically.

- **Example**

  ```python
  # Example 1: Download a file to the current directory
  download_from_google(
      file_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      file_name="example_file.txt"
  )

  # Example 2: Download a file to a specified directory
  download_from_google(
      file_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      file_name="example_file.txt",
      target="./downloads"
  )
  ```
