---
sidebar_position: 5
---

# 回測系統分析

## PythonTechAnalysis

當你下載完元富證券的 Python 模組後，使用以下指令安裝回測系統：

```powershell
pip install .\MasterLink_PythonAPI\Python_tech_analysis\tech_analysis_api_v2-0.0.5-py3-none-win_amd64.whl
```

:::tip
在我們使用時，該套件的版本為 `0.0.5`。
:::

## 官方技術文件

- [**元富證券-技術分析**](https://mlapi.masterlink.com.tw/web_api/service/document/python-analysis)
- [**官方程式範例：example.py**](https://github.com/DocsaidLab/AutoTraderX/blob/main/MasterLink_PythonAPI/Python_tech_analysis/example.py)

## 核心模組

同樣地，我們把元富證券的 Python 模組解壓縮之後，直接分析其核心模組，這個部分的程式碼出乎意料地很簡單，因為所有的功能全部都包裝在 `.dll` 檔案內，而 Python 模組只是一個簡單的介面而已。

### TechAnalysis

這段程式在一開始直接引入 `TechAnalysisAPI` 這個被包裝在 `.dll` 檔案內的物件。

接著定義了一些需要掛載進 `TechAnalysisAPI` 的事件，這個部分我們不會去修改，也不會去調用這些函數，因此先不管它。

- **`Login`**

  使用時，必須先透過這個函數來登入元富證券的 API。

- **`GetHisBS_Stock`**

  這個函數需要指定一檔股票，還有指定一個日期，接著它會回傳該股票在該日期的「每個 Tick」的成交明細。

- **`SubTA` 和 `UnSubTA`**

  這兩個函數允許使用者訂閱或取消訂閱特定的技術指標。

- **技術指標**

  透過 TechAnalysisAPI，此模組支援多種技術指標，包括：

  - SMA（簡單移動平均）
  - EMA（指數移動平均）
  - RSI（相對強弱指數）
  - MACD（移動平均收斂發散指標）
  - KD（隨機指標）
  - CDP（逆勢操作指標）
  - BBands（布林帶）

### Model

剛才也提到，這個模組中，只有兩個 `.py` 檔，其中一個是 `TechAnalysis`，另一個就是 `Model`。

其內容是定義了一系列的數據結構和類別，主要用於股市技術分析。

以下是對代碼中每個部分的解析和說明：

- **枚舉類型（Enum）**

  代碼定義了幾個枚舉類型，用於表示不同的技術指標類型、時間單位以及股價升跌狀態：

  - `eTA_Type`: 這個枚舉類型用來表示各種技術分析指標，如簡單移動平均（SMA）、權重移動平均（WMA）、指數移動平均（EMA）等。
  - `eNK_Kind`: 表示 K 線的時間範圍，如日線、1 分鐘線等。
  - `eRaiseFall`: 表示價格的升或跌。

- **數據結構（Data Classes）**

  這些數據類別提供了存儲股市交易數據的結構：

  - `TKBarRec`: 存儲 K 線數據的類別，包括日期、產品、時間序列、價格以及交易量等。
  - `TBSRec`: 存儲特定股票或商品的買賣記錄。
  - 各種技術分析指標類別（如`ta_sma`, `ta_ema`, `ta_wma`等），每個類別都包括相應的 K 線數據和計算得出的指標值。

- **技術分析指標類別**

  這些類別用於表示不同的技術分析指標，並將指標值與對應的 K 線數據結合起來。例如：

  - `ta_sma`、`ta_ema` 和 `ta_wma` 類別分別代表簡單移動平均、指數移動平均和權重移動平均。
  - `ta_sar` 類別包含附加信息，如停損點和升跌狀態。
  - `ta_rsi` 和 `ta_macd` 類別提供了相對強弱指數和移動平均收斂發散指數的相關數據。
  - `ta_kd`、`ta_cdp` 和 `ta_bbands` 類別分別表示隨機指標、逆勢操作指標和布林帶指標的數據。

- **`k_setting` 類別**

  這個類別用於設定技術分析的配置，包括產品 ID、時間範圍、技術分析指標類型和開始日期。

  此類別是技術分析設定的基礎，用於初始化和配置技術分析 API 的呼叫。
