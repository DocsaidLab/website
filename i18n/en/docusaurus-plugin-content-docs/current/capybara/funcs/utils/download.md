# download

## gen_download_cmd

> [gen_download_cmd(file_id: str, target: str) -> str](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/utils.py#L68)

- **Description**: Generates the command to download a file from Google Drive.

- **Parameters**

  - **file_id** (`str`): The file ID.
  - **target** (`str`): The target path to save the downloaded file.

- **Example**

  ```python
  import capybara as cb

  file_id = '1c1b9b1b0cdcwfjowief'
  target = 'example.txt'
  cmd = cb.gen_download_cmd(file_id, target)
  print(cmd)
  # >>> wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget
  # >>> --quiet
  # >>> --save-cookies /tmp/cookies.txt
  # >>> --keep-session-cookies
  # >>> --no-check-certificate 'https://docs.google.com/uc?export=download&id=1c1b9b1b0cdcwfjowief'
  # >>> -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1c1b9b1b0cdcwfjowief" -O example.txt && rm -rf /tmp/cookies.txt
  ```

## download_from_docsaid

> [download_from_docsaid(file_id: str, file_name: str, target: str) -> None](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/utils.py#L79)

- **Description**: Downloads data from Docsaid's private cloud.

- **Parameters**

  - **file_id** (`str`): The file ID.
  - **file_name** (`str`): The file name.
  - **target** (`str`): The target path to save the downloaded file.

- **Example**

  ```python
  import capybara as cb

  file_id = 'c1b9b1b0cdcwfjowief'
  file_name = 'example.txt'
  target = 'example.txt'
  cb.download_from_docsaid(file_id, file_name, target)
  ```
