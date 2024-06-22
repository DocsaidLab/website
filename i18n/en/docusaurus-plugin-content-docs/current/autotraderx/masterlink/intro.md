---
sidebar_position: 1
---

# Overview

Due to the lack of sufficient technical documentation provided by MasterLink's API, we need to deconstruct and analyze the source code ourselves.

During this analysis process, we encountered many unique coding practices, which served as a learning experience.

:::info
MasterLink's API is not an open-source project, so we cannot directly push modified code.

If we modify their released `.whl` files and republish them, it might violate their terms of use.

Therefore, instead of directly modifying their code, we will build wrappers around their implementations with our code.
:::

## Case Records

### Example 1

```python title="SolPYAPI/PY_Trade_package/MarketDataMart.py"
class MarketDataMart:

    #region System Event Notification
    OnSystemEvent: callable = None
    def Fire_OnSystemEvent(self, data: SystemEvent):
        if self.OnSystemEvent is None:
            return
        if self.OnSystemEvent is not None:
            self.OnSystemEvent(data)
    #endregion

    # ...omitted
```

In this snippet, `OnSystemEvent` is defined but not used, and almost every method has an externally declared variable that is not used.

We believe if there were only one or two instances, it might be an error, but since it's pervasive, it could be an internal specification that we have yet to understand.

### Example 2

```python title="SolPYAPI/PY_Trade_package/SolPYAPI_Model.py"
class TSolQuoteFutSet():

    N1: int = 1
    N2: int = 2
    N3: int = 3
    N4: int = 4
    N5: int = 5

    MarketPriceOrder_Buy: str = "999999999"
    """Market price buy order, represented by nine 9s"""
    MarketPriceOrder_Sell: str = "-999999999"
    "Market price sell order, represented by nine 9s"
    TryMark_Yes: str = "1"
    "Pre-market trial match"
    TryMark_No: str = "0"
    "Intraday trading"
    def __setattr__(self, *_):
        raise Exception("Tried to change the value of a constant")
```

In this module, strings are directly used as comments within the code.

### Example 3

```python title="SolPYAPI/PY_Trade_package/SolClientOB.py"
import datetime #line:1
from threading import Lock #line:2
from PY_Trade_package.SolLog import *#line:3
from PY_Trade_package.SolPYAPI_Model import *#line:4
from PY_Trade_package.Helper import MQCSHelper #line:5

# ...omitted
```

In this module, there are very precise comments.

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
# ...omitted
```

In a particular code file, we unexpectedly encountered obfuscated encryption methods.

Clearly, MasterLink considers the content of this file very important and has therefore obfuscated it.

In such cases, if you need to improve code readability, we recommend using the editor to directly replace strings, restoring the original strings. This process should only take a few minutes.
