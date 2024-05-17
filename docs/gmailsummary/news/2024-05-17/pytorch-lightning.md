# pytorch-lightning

## 2024-05-17 彙整報告

根據收到的電子郵件內容，我們可以將重要訊息歸納如下：



### 1. 錯誤修復

- 使用MLFlowLogger時出現的錯誤指出模組'pytorch_lightning.callbacks'缺少屬性'ProgressBarBase'，建議改用'ProgressBar'。這可能導致進度條顯示問題。建議開發者應該更新相應的程式碼以避免此錯誤。



### 2. 功能增加

- 使用者提到在Tensorboard中記錄超參數的問題，提及了`torch.utils.tensorboard.writer.SummaryWriter.add_hparams()`，但缺乏如何調用的清晰說明。使用者期望得到一個簡單的範例來使用HPARAMS標籤。建議提供使用HPARAMS標籤的範例程式碼，以便使用者能夠輕鬆地記錄超參數。



### 3. 討論的議題

- 使用者關注在分散式訓練中記錄日誌的問題，想知道在調用`self.log`時是否僅保存來自rank 0的結果，以及實際記錄是否僅在rank 0 GPU上執行。這涉及到分散式訓練中的日誌記錄機制，開發者可能需要檢查Lightning模組中的日誌記錄實現方式，以確保結果的正確性。



### 4. 特別提到的成就或挑戰

- 使用MLFlowLogger時出現的不一致指標圖問題，描述了概覽和詳細視圖中的圖表不一致情況。使用者懷疑這可能與`step`參數如何傳播到MLflow或如何計算`global_step`有關。提供的程式碼和圖片可能有助於釐清問題，開發者可能需要仔細檢查指標圖的生成過程以解決這個問題。



綜合以上內容，開發者應該注意修復MLFlowLogger的錯誤、提供HPARAMS標籤的範例程式碼、檢查分散式訓練中的日誌記錄機制，以及解決MLFlowLogger產生的不一致指標圖問題。這些措施將有助於提高代碼的穩定性和可靠性，確保訓練過程中的正確記錄和監控。



---



本日共彙整郵件： 19 封



以上報告由 OpenAI GPT-3.5 Turbo 模型自動生成。