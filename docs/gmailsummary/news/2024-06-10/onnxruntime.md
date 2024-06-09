# onnxruntime

## 2024-06-10 彙整報告

根據您提供的郵件內容，我們可以看到以下重要訊息和議題：



### 1. 錯誤修復與功能增加

   - 使用TensorRT 10.0 GA和ORT時出現問題，特別是在使用`-use_tensorrt_oss_parser`時。這表明在整合TensorRT和ONNX Runtime時可能存在一些兼容性問題，需要進行修復以確保順利運行。

   - 在使用ONNX Runtime 1.18.0時遇到問題，需要與TensorRT 10進行修復。這強調了軟體版本之間的相容性問題，特別是對於需要相互整合的庫或框架，如ONNX Runtime和TensorRT。

   - Microsoft.ML.OnnxRuntime.Gpu 1.18.0似乎需要CUDA 12，而不是CUDA 11.6。這指出了硬體和軟體之間的相容性要求，尤其是對於GPU加速的應用。



### 2. 討論的議題

   - 性能比較：討論了TensorRTExecutionProvider和CUDAExecutionProvider之間的性能比較，特別是在Faster-rcnn方面。這顯示了對於不同執行提供程序的性能評估和比較，有助於選擇最適合特定任務的執行提供程序。

   - GPU內存釋放：討論了如何在session運行後釋放GPU內存的問題。這是一個重要的議題，特別是對於長時間運行的模型或多個模型串行執行時，有效地管理GPU內存可以提高系統的效率和穩定性。

   - AddExternalInitializers問題：討論了使用AddExternalInitializers載入外部數據並輸出NaN的問題。這表明在模型初始化和數據載入過程中可能存在的錯誤，需要進一步調查和解決。

   - LayerNormalization問題：討論了在Dnnl執行提供程序上使用LayerNormalization可能引起的輸入副作用問題。這強調了在不同執行提供程序上執行特定操作時可能出現的問題，需要進行細致的測試和調試。



### 3. 成就與挑戰

   - DDS操作支持：提到了DDS操作的加速和ORT 1.18.0中對DDS操作的支持。這顯示了對於特定操作或功能的優化和支持，有助於提高模型執行效率和性能。

   - 相容性問題：提到了Microsoft.ML.OnnxRuntime.Gpu 1.18.0與NVIDIA CUDA 11.6之間的相容性問題。這強調了不同軟體版本或庫之間可能存在的相容性挑戰，需要確保所有組件能夠順利協同工作。



綜合以上訊息，我們可以看到在整合和使用TensorRT、ONNX Runtime以及相關GPU加速庫時，需要關注軟體版本相容性、性能比較、GPU內存管理和操作執行的問題。解決這些挑戰需要深入的技術了解和有效的協作，以確保模型的順利運行和最佳效能表現。



---



本日共彙整郵件： 8 封



以上報告由 OpenAI GPT-3.5 Turbo 模型自動生成。