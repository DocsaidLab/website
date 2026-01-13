# download_from_google

> [download_from_google(file_id: str, file_name: str, target: str | Path = ".") -> Path](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/utils.py)

- **說明**：從 Google Drive 下載檔案，並處理大型檔案的確認問題。

- **參數**

  - **file_id** (`str`)：要從 Google Drive 下載的檔案 ID。
  - **file_name** (`str`)：下載後保存的檔案名稱。
  - **target** (`str | Path`, 可選)：保存檔案的目標目錄，預設為當前目錄 `"."`。

- **傳回值**

  - **Path**：下載完成後的檔案路徑。

- **例外**

  - **Exception**：無法從回應中解析下載連結／確認參數時。
  - **RuntimeError**：寫檔流程失敗時。

- **備註**

  - 該函數處理小型和大型檔案。對於大型檔案，自動處理 Google 的確認憑證，繞過病毒掃描或檔案大小限制的警告。
  - 確保目標目錄存在，若不存在將自動創建。

- **範例**

  ```python
  from capybara.utils import download_from_google

  # 下載檔案到當前目錄
  path = download_from_google(
      file_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      file_name="example_file.txt"
  )

  # 下載檔案到指定目錄
  path = download_from_google(
      file_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      file_name="example_file.txt",
      target="./downloads"
  )
  ```
