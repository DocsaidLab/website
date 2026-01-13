---
sidebar_position: 1
---

# 介紹

Capybara 主要包括以下幾個部分（以程式碼結構為準）：

- **Vision**（`capybara.vision`）：影像／影片處理與 I/O。
- **Structures**（`capybara.structures`）：`Box/Boxes`、`Polygon/Polygons`、`Keypoints` 等幾何結構。
- **Runtime**（`capybara.runtime`）：推論 runtime / backend 的註冊表與選擇邏輯。
- **Inference engines（可選）**：
  - `capybara.onnxengine`（ONNXRuntime）
  - `capybara.openvinoengine`（OpenVINO）
  - `capybara.torchengine`（TorchScript）
- **Utils**（`capybara.utils`）：輔助工具（路徑、下載、時間、設定容器等）。
- **Extras（可選）**：
  - `visualization`：繪圖工具（`capybara.vision.visualization`）
  - `ipcam`：簡易 Web demo（`capybara.vision.ipcam`）
  - `system`：系統資訊工具（`capybara.utils.system_info`）

## Vision

Vision 模塊專注於處理圖像與影片資料，提供豐富的電腦視覺工具。

目錄結構：

```
vision
├── functionals.py       # 提供基本的影像處理函數，例如濾波、變換等
├── geometric.py         # 幾何處理相關函數，例如旋轉、縮放
├── improc.py            # 圖像處理核心邏輯
├── morphology.py        # 提供形態學處理，例如膨脹、腐蝕
├── videotools           # 影片工具相關模塊
├── ipcam                # （需 extra: ipcam）網路攝影機 demo
└── visualization        # （需 extra: visualization）繪圖與可視化
```

主要功能：

- 圖像與影片的讀取、處理及視覺化。
- 支持多種格式與來源（本地檔案、影片抽幀、IPCam demo 等）。

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

## Runtime / Inference engines（可選）

推論相關功能位於獨立模組中，並透過 `capybara.runtime` 提供 runtime/backend 的統一描述。

注意：推論後端為可選依賴，請先依需求安裝 extras（例如 `capybara-docsaid[onnxruntime]`）。

### capybara.runtime

- 定義 `Runtime` / `Backend`，並提供 `auto_backend_name()` 等選擇邏輯。

### capybara.onnxengine

目錄結構：

```
onnxengine
├── engine.py            # 核心推理邏輯
├── __init__.py          # 初始化文件
├── metadata.py          # 模型元數據管理
└── utils.py             # ONNX 解析與輔助工具
```

主要功能：

- 支持 ONNX 模型的載入與推理。

### capybara.openvinoengine

- OpenVINO 推論封裝（同步推論 + 可選 async queue）。

### capybara.torchengine

- TorchScript 推論封裝（支援簡單的 dtype / device 正規化）。

## Utils

Utils 模塊包含各種輔助工具函數，範圍廣泛。

目錄結構：

```
utils
├── custom_path.py       # 自定義路徑操作
├── custom_tqdm.py       # 進度條工具
├── files_utils.py       # 文件處理函數
├── powerdict.py         # 強化版字典操作
├── system_info.py       # （需 extra: system）系統資訊檢測
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

具體使用方式請參考後續 API 文件與範例；若遇到「無法 import」的情況，優先確認是否需要安裝對應的 extras。
