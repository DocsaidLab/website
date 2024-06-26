# onnxruntime

## 2024-06-05 彙整報告

根據收到的電子郵件內容，以下是一些重要訊息的提取和梳理：



1. **修復和功能增加**：

   - 修復了Gradle語法的問題，主要涉及修復了java/build.gradle中的deprecated Gradle語法，以提高代碼的可讀性和效率。

   - 新增了阻塞量化功能到DequantizeLinear操作核心，這項功能已經合併到主分支，並提到了更新代碼以支持多線程。

   - 更新了tensorprotoutils.h中的函數，使用了std::filesystem::path，這樣的更新可能提高了文件路徑操作的效率和可靠性。

   - 移除了Gradle cmakeCheck中的failOnStderr，這可能是為了消除不必要的錯誤檢查，以確保Gradle構建流程的順利執行。



2. **討論的議題**：

   - 討論了支援CUDA 12的cudnn9問題，提到了Azure提供的網格驅動程式版本以及對應的CUDA版本，這可能是為了確保在最新的CUDA環境下能夠正確運行。

   - 討論了Attention中的整數溢出問題，特別針對在32位和64位平台上指針大小的問題進行了深入探討，以確保代碼的穩定性和可靠性。

   - 討論了INT8 GEMM部分的代碼改進，詢問是否可以合併，這可能是為了提高INT8計算性能和精度。



3. **特別提到的成就或挑戰**：

   - 提到了EfficientNMS_TRT缺少屬性class_agnostic的問題，並感謝提供實驗建議，這顯示了團隊對於用戶反饋的積極回應和解決問題的態度。



綜合以上內容，可以看出團隊在持續努力修復代碼中的問題、增加新功能以提高性能、討論和解決特定議題，以及積極回應用戶反饋和挑戰。這些努力和成就表明了團隊對於代碼品質和功能性的關注，以確保項目的成功和持續發展。



在工程細節方面，Gradle語法的修復可能涉及到Gradle構建工具的配置和語法規範，阻塞量化功能的新增可能牽涉到深度學習模型的優化和加速，而更新函數中使用std::filesystem::path可能涉及到文件系統操作的改進和最佳實踐。討論中的整數溢出問題和INT8 GEMM部分的代碼改進可能需要對代碼進行細緻的分析和優化，以確保數據處理的正確性和效率性。



總的來說，團隊在不斷努力解決問題、改進功能和討論技術議題的過程中展現了專業和承諾，這將有助於提高項目的質量和用戶滿意度。



---



本日共彙整郵件： 142 封



以上報告由 OpenAI GPT-3.5 Turbo 模型自動生成。