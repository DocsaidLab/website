---
sidebar_position: 1
---

# 概要

元富証券が提供する API には十分な技術文書がないため、私たちは自分でソースコードを解析し、分析を行わなければなりませんでした。

その分析の過程で、非常に特別な書き方をいくつも見つけることができ、学びの過程でもあります。

:::info
元富証券の API はオープンソースプロジェクトではないため、私たちは直接変更したコードをプッシュすることはできません。

もし、彼らが公開した whl ファイルを直接変更し、それを再度公開することは、彼らの利用規約に違反する可能性があります。

したがって、私たちは彼らのコードを直接変更するのではなく、彼らの実装に基づき、私たち自身のコードでラップしています。
:::

## ケース記録

### Example 1

```python title="SolPYAPI/PY_Trade_package/MarketDataMart.py"
class MarketDataMart:

    #region システムイベント通知
    OnSystemEvent:callable = None
    def Fire_OnSystemEvent(self, data:SystemEvent):
        if self.OnSystemEvent == None:
            return
        if self.OnSystemEvent != None:
            self.OnSystemEvent(data)
    #endregion

    # ...以下省略
```

このコードでは、`OnSystemEvent`が定義されているのに使用されておらず、ほとんどすべてのメソッドには使用されていない変数が宣言されています。

私たちは、もし一つや二つならば書き間違いかもしれませんが、どこにでもある場合、それは彼らの内部的な仕様であり、私たちがまだその使い方を理解していないだけかもしれません。

### Example 2

```python title="SolPYAPI/PY_Trade_package/SolPYAPI_Model.py"
class TSolQuoteFutSet():

    N1:int = 1
    N2:int = 2
    N3:int = 3
    N4:int = 4
    N5:int = 5

    MarketPriceOrder_Buy:str = "999999999"
    """市場価格で買い注文を出す、相場は9個の9で表現"""
    MarketPriceOrder_Sell:str = "-999999999"
    "市場価格で売り注文を出す、相場は9個の9で表現"
    TryMark_Yes:str = "1"
    "前場試撮"
    TryMark_No:str = "0"
    "後場取引"
    def __setattr__(self, *_):
        raise Exception("定数の値を変更しようとしました")
```

モジュール内では、多くのコードの中で、文字列がそのままコメントとして使われていることがわかります。

### Example 3

```python title="SolPYAPI/PY_Trade_package/SolClientOB.py"
import datetime #line:1
from threading import Lock #line:2
from PY_Trade_package.SolLog import *#line:3
from PY_Trade_package.SolPYAPI_Model import *#line:4
from PY_Trade_package.Helper import MQCSHelper #line:5

# ...以下省略
```

モジュール内には、非常に正確なコメントがたくさんあります。

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

あるプログラムファイルには、突然プログラムコードが難読化されている手法が現れました。

元富証券はこのファイルの内容が非常に重要だと考え、このファイルに対して難読化を行ったようです。

このような場合、プログラムの可読性を高める必要がある場合は、エディタ内で文字列を直接置き換え、元の文字列を取り戻すことをお勧めします。この過程は数分程度の時間を要します。
