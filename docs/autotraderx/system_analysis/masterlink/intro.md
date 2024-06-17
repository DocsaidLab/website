---
sidebar_position: 1
---

# 概述

由於元富證券所提供的 API 並沒有足夠的技術文件，因此我們必須自己拆解原始碼，並進行分析。

在分析的過程中看到很多很特別的寫法，也算是一種學習的過程。

:::info
元富證券的 API 不是開源專案，因此我們無法直接推送修改後的程式碼。

如果我們直接拿他們發布的 whl 檔案進行修改後，另外重新發布的話，可能會違反他們的使用條款。

因此我們不會直接修改他們的程式碼，而是基於他們的實作，再透過我們自己的程式碼進行封裝。
:::

## 案例紀錄

### Example 1

```python title="SolPYAPI/PY_Trade_package/MarketDataMart.py"
class MarketDataMart:

    #region 系統訊息通知
    OnSystemEvent:callable = None
    def Fire_OnSystemEvent(self, data:SystemEvent):
        if self.OnSystemEvent == None:
            return
        if self.OnSystemEvent != None:
            self.OnSystemEvent(data)
    #endregion

    # ...以下省略
```

這段程式碼中，`OnSystemEvent` 被定義了之後卻沒有被使用，而且幾乎每個方法都有一個宣告在外面卻沒有被使用的變數。

我們認為如果只有一兩個，那可能是寫錯，但到處都是的話，可能是他們某種內部的規格，只是我們尚未理解這個用法。

### Example 2

```python title="SolPYAPI/PY_Trade_package/SolPYAPI_Model.py"
class TSolQuoteFutSet():

    N1:int = 1
    N2:int = 2
    N3:int = 3
    N4:int = 4
    N5:int = 5

    MarketPriceOrder_Buy:str = "999999999"
    """市價買進委託,行情以9個9來表達"""
    MarketPriceOrder_Sell:str = "-999999999"
    "市價賣出委託,行情以9個9來表達"
    TryMark_Yes:str = "1"
    "盤前試撮"
    TryMark_No:str = "0"
    "盤中交易"
    def __setattr__(self, *_):
        raise Exception("Tried to change the value of a constant")
```

模組內有大量的程式碼中，字串會直接被當成註解使用。

### Example 3

```python title="SolPYAPI/PY_Trade_package/SolClientOB.py"
import datetime #line:1
from threading import Lock #line:2
from PY_Trade_package.SolLog import *#line:3
from PY_Trade_package.SolPYAPI_Model import *#line:4
from PY_Trade_package.Helper import MQCSHelper #line:5

# ...以下省略
```

模組內有大量的程式碼中，存在很多非常精準的註解。

### Example 4

```python title="SolPYAPI/PY_Trade_package/SolClientOB.py"
class SolClient :#line:93
    host =''#line:95
    vpn =''#line:96
    username =''#line:97
    password =''#line:98
    cacheName ='dc01'#line:99
    clientName =''#line:100
    cacheRequestTimeoutInMsecs =50000 #line:101
    requestTimeoutInMsecs =50000 #line:102
    session =None #line:103
    context =None #line:104
    Jan1st1970 =datetime .datetime (1970 ,1 ,1 ,0 ,0 ,0 ,tzinfo =datetime .timezone .utc )#line:106
    def __init__ (O00000O000O0O000O ,OO0000O0OOOOOOO0O :str ,OOOO0OO00O00OOO00 :str ,O00O0OO0OO00OOOOO :str ,OO000OO0O0O00O0OO :str ,O0OOO0O0000OO000O :SolaceLog ,clientName :str =""):#line:107
# ...以下省略
```

在某個程式檔案中，猝不及防地出現了程式碼混淆的加密手法。

顯然元富證券認為這個檔案的內容非常重要，因此對這個檔案進行了混淆加密。

在這種情況下，如果你需要提高程式的可讀性，我們建議你可以在編輯器內直接進行字串取代，找回原本的字串，這個過程大概會耽誤你幾分鐘的時間。
