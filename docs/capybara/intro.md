---
sidebar_position: 1
---

# 介紹

Capybara 主要包括以下幾個部分：

- **Vision**：包括與電腦視覺相關的功能，如圖像和影片處理。
- **Structures**：用於處理結構化數據的模塊，例如 BoundingBox 和 Polygon。
- **ONNXEngine**：提供 ONNX 推理的功能，支持 ONNX 格式模型。
- **Utils**：包含各種工具函數，提供系統資訊、文件處理等輔助功能。
- **Tests**：測試文件，用於驗證各類函數的功能。

## Vision

Vision 模塊專注於處理圖像與影片資料，提供豐富的電腦視覺工具。

目錄結構：

```
vision
├── functionals.py       # 提供基本的影像處理函數，例如濾波、變換等
├── geometric.py         # 幾何處理相關函數，例如旋轉、縮放
├── improc.py            # 圖像處理核心邏輯
├── ipcam                # 用於處理網路攝影機流的模塊
├── morphology.py        # 提供形態學處理，例如膨脹、腐蝕
├── videotools           # 影片工具相關模塊
└── visualization        # 提供視覺化工具，例如畫框、標註
```

主要功能：

- 圖像與影片的讀取、處理及視覺化。
- 支持多種格式及來源（如本地文件、網絡攝影機）。

## Structures

Structures 模塊負責處理結構化數據，常用於電腦視覺與數據分析場景。

目錄結構：

```
structures
├── functionals.py       # 相關功能函數
├── boxes.py             # Box、Boxes 資料結構
├── keypoints.py         # Keypoints 資料結構
└── polygons.py          # Polygon、Polygons 資料結構
```

主要功能：

- 提供 Box、Keypoints、Polygon 等結構化數據處理。
- 支持多種操作如交集、並集、縮放等。

## ONNXEngine

ONNXEngine 模塊提供與 ONNX 格式模型推理相關的功能。

目錄結構：

```
onnxengine
├── engine.py            # 核心推理邏輯
├── __init__.py          # 初始化文件
└── metadata.py          # 模型元數據管理
```

主要功能：

- 支持 ONNX 模型的載入與推理。

## Utils

Utils 模塊包含各種輔助工具函數，範圍廣泛。

目錄結構：

```
utils
├── custom_path.py       # 自定義路徑操作
├── custom_tqdm.py       # 進度條工具
├── files_utils.py       # 文件處理函數
├── powerdict.py         # 強化版字典操作
├── system_info.py       # 系統資訊檢測
├── time.py              # 時間處理工具
└── utils.py             # 通用工具函數
```

主要功能：

- 文件處理與系統資訊檢測。
- 提供自定義工具如進度條與強化版字典。

## Tests

Tests 模塊用於驗證系統的功能是否正確。

主要功能：

- 包含各模塊的單元測試。
- 提供快速回歸與功能驗證。

---

以上是 Capybara 各模塊的初步介紹。

具體使用方式可以繼續往後閱讀相應的 API 文件及範例程式碼。
