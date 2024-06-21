---
sidebar_position: 8
---

# 回測系統

分析完元富證券的回測系統 Python API 之後，我們就可以基於自己的需求，開發一個回測系統。

## 登入帳號

你可以直接把帳號密碼寫在類別的輸入中，也可以參考我們的寫法：使用一個 yaml 檔案來儲存帳號資訊。

參數檔案中，必須有帳號密碼和帳號號碼，這樣才能順利登入元富證券的帳號。

接著從 `autotraderx` 中 import `BackTesting` 類別：

```python
from autotraderx import load_yaml
from autotraderx.masterlink import BackTesting

# Load account infos
cfg = load_yaml(DIR / "account.yaml")

# Login account
handler = BackTesting(
    user=cfg["user"],
    password=cfg["password"],
)
```

## 訂閱指標

我們在測試官方提供的範例程式碼時，發現在訂閱指標的時候，有一個「非常久」的不反應期。

以下是由官方提供的範例程式碼：

```python
ta = TechAnalysis(...)

opt = input("1: 指標\n2: 歷史成交\n> ")
if opt == "1":
    k_config = option()
    ta.SubTA(k_config)
    input("running...\n")
    ta.UnSubTA(k_config)
```

執行之後，我們卡在 `running...` 的畫面，大約五分鐘，最後是因為我們失去耐心而強制結束程式。

此外，技術指標的部分大多都是由價量資訊計算而得，所以我們自己取回價量資料，計算指標什麼的，自己算比較快。

所以我們不實作這一部分的功能。

## 歷史成交

登入帳號之後，就可以使用我們包裝的 `get_data` 函數，來取得資料。

```python
data = handler.get_data(
    prod_id="2330",
    date="20240102",
)
```

其中，`prod_id` 是股票代碼，`date` 是日期，格式為 `YYYYMMDD`。

以上的程式執行之後，會回傳一個 List[Dict] 結構：

```json
[
    ...以上省略...
    {
        "成交價格": 590.0,
        "成交時間": 132459.132661,
        "成交量": 1,
        "股票代號": "2330",
        "試搓": False,
        "買賣": 2
    },
    {
        "成交價格": 593.0,
        "成交時間": 133000.0,
        "成交量": 3704,
        "股票代號": "2330",
        "試搓": False,
        "買賣": 2
    }
]
```

經過我們的觀察，發現元富證券提供的 API 中，有幾個特性：

1. **`試搓`** 這個欄位，代表這筆成交是否為試搓，但是我們發現全部的資料都是 `False`。（？？？）
2. **`買賣`** 這個欄位，我們不知道代表的意義是什麼，從程式碼中，沒有找到相關的註解說明。
3. 我們試著查詢過去的資料，發現在 2022 年 4 月中旬以後，才有資料可以用。

不過兩年多的資料，就我們這種小規模的使用者來說，應該是夠用了。

## 後續工作

我們預計先把可以查到的股票成交資訊資料取回來，並在自己建置一個資料庫系統。

後續要計算技術指標或是回測交易策略的時候，就可以直接從我們的資料庫中取得資料，又快又方便。

至於更久遠的資料，等我們有更多的需求再說吧。
