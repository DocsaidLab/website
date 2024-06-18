---
sidebar_position: 6
---

# 下單系統

分析完元富證券的下單系統 Python API 之後，我們就可以基於自己的需求，開發一個下單系統。

在大多數的情境下，我們只會單獨操作一個證券帳戶，所以我們目前的實作是基於單一帳號的情境。

:::warning
我們不推薦任何股票，所有關於證券戶內的股票資訊都會以馬賽克處理。
:::

## 登入帳號

你可以直接把帳號密碼寫在類別的輸入中，也可以參考我們的寫法：使用一個 yaml 檔案來儲存帳號資訊。

參數檔案中，必須有帳號密碼和帳號號碼，這樣才能順利登入元富證券的帳號。

```python
from autotraderx import Trader, load_yaml

# Load account infos
cfg = load_yaml(DIR / "account.yaml")

# Login account
account = Trader(
    user=cfg["user"],
    password=cfg["password"],
    account_number=str(cfg["account_number"]),
    is_sim=False,
    is_force=True,
    is_event=False,
)

account.login()
# Do something
account.stop()
```

## 查詢庫存

登入帳號，呼叫 `get_inventory` 函數，就可以查詢目前的庫存狀況。

查詢結果會直接顯示在命令列中，我們可以看到目前的庫存狀況，如果要取得資訊做進一步地使用，可以直接使用回傳值。

```python
data = account.get_inventory()
```

![查詢庫存](./img/get_inventory.jpg)

其中，`data` 的輸出格式為一個字典，內容像是：

```python
{
    '2002': {
        '股票': '中鋼',
        '融券庫存（張）': '0',
        '融資庫存（張）': '0',
        '集保庫存（張）': '1',
        '零股庫存（股）': '0'
    },
    '2330': {
        '股票': '台積電',
        '融券庫存（張）': '0',
        '融資庫存（張）': '0',
        '集保庫存（張）': '1',
        '零股庫存（股）': '0'
    },
    # ...以下省略
}
```

如果不需要顯示在命令列，可以在初始化 `Trader` 時，設定 `verbose` 參數為 `False`。

```python
account = Trader(
    verbose=False
)
```

## 查詢委託資訊

登入帳號，呼叫 `get_order_report` 函數，就可以查詢目前的所有委託資訊。

查詢結果會直接顯示在命令列中，我們可以看到目前的所有委託資訊，如果要取得資訊做進一步地使用，可以直接使用回傳值。

![查詢委託資訊](./img/get_order_report.jpg)

```python
data = account.get_order_report()
```

其中，`data` 的輸出格式為 List\[Dict\]，內容像是：

```python
[
    {
        '委託價': '13.95',
        '委託方式(價格)': '限價單',
        '委託方式(效期)': '當日有效',
        '委託時間': '08:31:32.032000',
        '委託書號': 'i0040',
        '委託量': '4000',
        '成交價': '',
        '成交時間': '',
        '成交量': '',
        '狀態': '',
        '股票': '馬賽克',
        '股票代號': '馬賽克',
        '訊息': '',
        '買賣別': 'Sell',
        '類型': '委託訂單'
    },
    # ...以下省略
]
```

## 查詢成交資訊

登入帳號，呼叫 `get_trade_report` 函數，就可以查詢目前的所有成交資訊。

查詢結果會直接顯示在命令列中，我們可以看到目前的所有成交資訊，如果要取得資訊做進一步地使用，可以直接使用回傳值。

![查詢成交資訊](./img/get_trade_report.jpg)

```python
data = account.get_trade_report()
```

其中，`data` 的輸出格式為 List\[Dict\]，內容像是：

```python
[
    {
        '委託價': '13.95',
        '委託方式(價格)': '限價單',
        '委託方式(效期)': '當日有效',
        '委託時間': '08:31:32.032000',
        '委託書號': 'i0040',
        '委託量': '4000',
        '成交價': '',
        '成交時間': '09:00:11.609000',
        '成交量': '4000',
        '狀態': '111)全部成交',
        '股票': '馬賽克',
        '股票代號': '馬賽克',
        '訊息': '',
        '買賣別': 'Sell',
        '類型': '委託訂單'
    },
    # ...以下省略
]
```

## 下單定義型別

在下單的過程中，元富證券定義了幾個常數，我們需要先了解這些常數的意義。

### OrderType

```python
# 委託方式(效期)
class OrderType(str, Enum):
    # 當日有效
    ROD = "R"
    # 立即成交，否則取消
    IOC = "I"
    # 立即全部成交，否則取消
    FOK = "F"
```

### PriceType

```python
# 委託方式(價格)
class PriceType(str, Enum):
    # 限價單
    LMT = "L"
    # 市價單
    MKT = "M"
```

### TradingType

```python
# 委託別
class TradingType(str, Enum):
    # 集保
    CUSTODY = "G"
```

### TradingUnit

```python
# 交易單位
class TradingUnit(int,Enum):
    COMMON = 1000
    ODD = 1
```

### TradingSession

```python
# 交易時段
class TradingSession(str, Enum):
    # 普通
    NORMAL = "N"
    # 盤後
    FIXED_NORMAL = "F"
    # 盤中零股
    ODD = "R"
    # 盤後零股
    FIXED_ODD = "L"
```

### Side

```python
# 買賣別
class Side(str, Enum):
    # 買進
    Buy = "B"
    # 賣出
    Sell = "S"
```

## 下單買進

登入帳號，呼叫 `buy` 函數，就可以下買進單。

例如，買進台積電（股票代號：2330） 1 張，價格為 500 元。

預設下單方式：

- 價格類型：OrderType.KMT, 限價單
- 委託類型：PriceType.ROD, 當日有效
- 交易時段：TradingSession.NORMAL, 一般交易時段
- 交易單位：TradingUnit.COMMON, 一般交易單位

```python
account.buy(symbol="2330", qty=1, price=500)
```

## 下單賣出

登入帳號，呼叫 `sell` 函數，就可以下賣出單。

例如，賣出台積電（股票代號：2330） 1 張，價格為 500 元。

預設下單方式：

- 價格類型：OrderType.KMT, 限價單
- 委託類型：PriceType.ROD, 當日有效
- 交易時段：TradingSession.NORMAL, 一般交易時段
- 交易單位：TradingUnit.COMMON, 一般交易單位

```python
account.sell(symbol="2330", qty=1, price=500)
```

## 自定義下單方式

登入帳號，呼叫 `set_order` 函數，就可以下自定義單。

以下展示了 `set_order` 函數的定義，依照你的需求，自行設定下單方式。

```python
 def set_order(
        self,
        symbol: str,    # 股票代號
        side: Side,     # 買賣別
        qty: int,       # 委託股數
        price: float,   # 委託價格
        order_type: OrderType = OrderType.ROD,  # 委託類型
        price_type: PriceType = PriceType.MKT,  # 價格類型
        trading_session: TradingSession = TradingSession.NORMAL,  # 交易時段
        trading_unit: TradingUnit = TradingUnit.COMMON,  # 交易單位
    ):
        self.api.ReqBasic(symbol)

        order = Order(
            tradingSession=trading_session,
            side=side,
            symbol=symbol,
            priceType=price_type,
            price=str(price),
            tradingUnit=trading_unit,
            qty=str(qty),
            orderType=order_type,
            tradingAccount=self.account_number,
            userDef=''
        )
        rc = self.api.NewOrder(order)
        if rc == RCode.OK:
            print(u'已送出委託')
        else:
            print(u'下單失敗! 請再次執行程式，依據回報資料修正輸入')
```

:::warning
在 `set_order` 函數中，若要買進 1 張，則 `qty` 參數應該為 1000。
:::

例如：買進台積電（股票代號：2330） 1 張，價格為 500 元，委託類型為立即全部成交，價格類型為市價單。

```python
account.set_order(
    symbol="2330",
    side=Side.Buy,
    qty=1000,
    price=500,
    order_type=OrderType.FOK,
    price_type=PriceType.MKT
)
```

## 改價

必須先找到要改價的「委託單號」，然後呼叫 `change_price` 函數，就可以改價。

例如，台積電（股票代號：2330）改價，新價格為 600 元。

```python
account.change_price(order_number="i0041", mod_price=600)
```

## 改量

必須先找到要改量的「委託單號」，然後呼叫 `change_qty` 函數，就可以改量。

例如，台積電（股票代號：2330）改量，從 1 張改為 2 張。

```python
account.change_qty(order_number="i0041", mod_qty=2000)
```

## 刪單

使用改量的方式，將「委託量」改為 0，就是刪單。

## 其他功能

原始的 API 還有查詢資券餘額的功能，由於我們目前手邊的帳號沒有開通相關權限，所以我們暫時無法針對這些功能進行開發測試。

此外，我們沒有在元富證券的 API 中找到「查詢帳戶庫存成交價的資訊」，於是嘗試從程式碼下去追蹤，我們發現最後的資料填充層被封裝在 `.dll` 檔案內，如果要取得這些資料，還要再對 `.dll` 進行反編譯後，再進行資料的解析......，這個工作量對於我們來說有點大！

總之，我們目前無法提供「計算庫存均價」的服務，祈禱未來元富證券可以把這個功能開放出來。

:::tip
如果你知道該如何取得「庫存均價」，也就是「每張庫存的成交價」的資訊，拜託告訴我們！🙏 🙏 🙏
:::
