# onnxruntime

## 2024-04-29 彙整報告

根據您提供的電子郵件內容，這裡有一些重要的關鍵訊息：



### 功能增加：

1. **GroupQueryAttention 支持**:

   - 在 PR #20237 中，@axinging 在 [js/webgpu] 中新增了對 GroupQueryAttention 的支持。這表示現在可以在相關的領域中使用 GroupQueryAttention 功能，這將有助於提高模型的效能和準確性。

   

2. **SparseAttention operator for Phi-3-small**:

   - 另一方面，在 PR #20216 中，@tianleiwu 在 CUDA 中添加了 SparseAttention operator for Phi-3-small。這個新的 operator 將為 CUDA 帶來更多的功能和靈活性，有助於處理稀疏注意力機制的模型。



### 錯誤修復：

1. **[[maybe_unused]] 移除**:

   - 在 Issue #20233 中，建議編輯特定行來移除 `[[maybe_unused]]`，這可能是為了解決編譯或運行時的錯誤，確保程式碼的正確性和穩定性。



### 更新與討論：

1. **程式碼格式化**:

   - 在 PR #20165 中，@fs-eire 對程式碼進行了格式化。這有助於提高程式碼的可讀性和維護性，使團隊成員更容易理解和修改程式碼。



2. **README.md 更新**:

   - 另外，在 PR #20492 中更新了 /js/web/ 中的 README.md，包括相容性表格和連結到 onnxruntime.ai。這將幫助使用者更好地了解程式庫的相容性和相關資源。



3. **WebGPU 支援**:

   - 關於 Issue #20465，討論了 web "Get started" 文件和 "js" 資料夾之間的矛盾，並指出 WebGPU 仍被視為實驗性功能，預計在 v1.19.0 版本中正式推出。這表明團隊正在積極處理和改進 WebGPU 的支援，以提供更好的用戶體驗。



### 成就討論：

1. **json5 版本更新**:

   - 在 PR #14111 中，討論了在 /js/react_native 中 Bump json5 from 1.0.1 to 1.0.2 的相關內容。這可能涉及到對程式庫依賴項的更新和相容性問題，這些討論有助於確保程式庫的穩定性和功能性。



以上是從您提供的電子郵件內容中提取的重要訊息。這些功能增加、錯誤修復、更新和討論的議題都顯示了團隊在持續改進和發展程式庫的努力。如果您需要進一步了解任何特定主題或有其他問題，請隨時告訴我。



---



本日共彙整郵件： 59 封



以上報告由 OpenAI GPT-3.5 Turbo 模型自動生成。