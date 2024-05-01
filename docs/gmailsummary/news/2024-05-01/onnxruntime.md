# onnxruntime

## 2024-05-01 彙整報告

根據您提供的多封電子郵件內容，這些是來自Microsoft Onnxruntime專案的一些重要訊息提取：



### 1. **錯誤修復**:

   - 在不同郵件中提到了多倥錯誤修復的情況，包括從protobuf中刪除使用'mach_absolute_time' iOS API的修補程式，以避免添加隱私清單，以及修復將onnxruntime資料夾放在包含其他Unicode字符的路徑下導致初始化提供者橋接失敗的問題。這些修復顯示了團隊對於提高代碼穩定性和可靠性的努力。



### 2. **功能增加**:

   - 有多個Pull Request提到了功能增加的情況，例如支持HardSigmoid的Pull Request以及為添加tensor v2支持以解鎖從QNN v2.21生成的上下文二進制文件的推理。這些功能增加表明了團隊致力於不斷擴展Onnxruntime的功能和性能。



### 3. **討論的議題**:

   - 在討論中涉及了各種議題，如在Flutter中放大模型輸出圖像時出現問題、Training mode不支持BN opset 14的問題、以及在Tensor轉換中使用std::move的問題。這些討論反映了團隊對於解決挑戰和改進功能的積極參與。



### 4. **特別提到的成就或挑戰**:

   - 一些Pull Request已經合併，包括更新了openai-whisper版本在requirements.txt中、更新了Qnn nuget包等。這些成就突顯了團隊在維護和改進Onnxruntime專案方面的努力。



綜合來看，這些電子郵件反映了Microsoft Onnxruntime專案團隊在持續進行錯誤修復、功能增加和討論議題的過程中取得的進展。團隊不僅致力於提高代碼質量和性能，還積極參與解決挑戰和討論新功能的實現。這種持續的努力和專注將有助於推動Onnxruntime專案的發展，並為用戶提供更好的服務和功能。



在這些訊息中，一些專有名詞如ONNX、QNN等可能需要進一步解釋。ONNX是一個開放的標準，用於表示機器學習模型，而QNN則可能指代Quantized Neural Networks，用於優化和加速神經網絡的技術。這些技術和標準在深度學習和機器學習領域中起著重要作用。



---



本日共彙整郵件： 158 封



以上報告由 OpenAI GPT-3.5 Turbo 模型自動生成。