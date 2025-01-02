# download

## download_from_google

> [download_from_google(file_id: str, file_name: str, target: str = ".") -> None](https://github.com/DocsaidLab/Capybara/blob/c83d96363a3de3686a98dd7b558a168f08b9bc97/capybara/utils/utils.py#L70)

- **說明**：從 Google Drive 下載檔案，並處理大型檔案的確認問題。

- **參數**

  - **file_id** (`str`)：要從 Google Drive 下載的檔案 ID。
  - **file_name** (`str`)：下載後保存的檔案名稱。
  - **target** (`str`, 可選)：保存檔案的目標目錄，預設為當前目錄 `"."`。

- **例外**

  - **Exception**：如果下載失敗或檔案無法創建，則引發異常。

- **備註**

  - 該函數處理小型和大型檔案。對於大型檔案，自動處理 Google 的確認憑證，繞過病毒掃描或檔案大小限制的警告。
  - 確保目標目錄存在，若不存在將自動創建。

- **範例**

  ```python
  # 範例 1：下載檔案到當前目錄
  download_from_google(
      file_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      file_name="example_file.txt"
  )

  # 範例 2：下載檔案到指定目錄
  download_from_google(
      file_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      file_name="example_file.txt",
      target="./downloads"
  )
  ```
