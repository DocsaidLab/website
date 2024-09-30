---
sidebar_position: 4
---

# 下單系統分析

## MasterTradePy

當你下載完元富證券的 Python 模組後，使用以下指令安裝下單系統：

```powershell
pip install .\MasterLink_PythonAPI\MasterTradePy\MasterTradePy\64bit\MasterTradePy-0.0.23-py3-none-win_amd64.whl
```

:::tip
在我們使用時，該套件的版本為 `0.0.23`。
:::

## 官方技術文件

- [**元富證券-下單 API**](https://mlapi.masterlink.com.tw/web_api/service/document/python-trade)
- [**官方程式範例：sample.py**](https://github.com/DocsaidLab/AutoTraderX/blob/main/MasterLink_PythonAPI/MasterTradePy/MasterTradePy/64bit/sample.py)

## 核心模組

我們把元富證券的 Python 模組拆解成以下幾個核心模組：

### Solace

:::tip
Solace 是一家提供高效能消息傳遞中介軟體和相關服務的公司，其技術特別適用於實時資料流和事件驅動的架構。
:::

我們會在程式中看到大量的 `Solace` 類別，這些類別是用來與 Solace 消息代理進行通信的。

1. **Messaging Service**

   - **MessagingService**: 這是消息服務的主要介面，用於配置和建立與 Solace 消息代理的連接。
   - **ReconnectionListener**: 這些介面用於處理連接中斷和重連的事件。
   - **ReconnectionAttemptListener**: 這些介面用於處理連接中斷和重連的事件。
   - **ServiceInterruptionListener**: 這些介面用於處理連接中斷和重連的事件。
   - **RetryStrategy**: 這個介面用於定義消息重傳的策略。
   - **ServiceEvent**: 處理有關服務生命週期的事件。

2. **Topic Subscription**

   - **TopicSubscription**: 用於訂閱特定主題的消息。

3. **Message Receiver and Publisher**

   - **MessageHandler**: 這些用於處理接收到的消息。
   - **InboundMessage**: 這些用於處理接收到的消息。
   - **DirectMessageReceiver**: 直接從消息代理接收消息的接收器。
   - **OutboundMessage**: 表示要發送的消息。
   - **RequestReplyMessagePublisher**: 用於發布請求並接收回應的消息發布者。
   - **PublishFailureListener**: 這些用於處理消息發布失敗的事件。
   - **FailedPublishEvent**: 這些用於處理消息發布失敗的事件。

4. **Topic and Caching**

   - **Topic**: 表示消息主題，可以是動態的或固定的。
   - **CachedMessageSubscriptionRequest**: 處理對緩存消息的訂閱請求。
   - **CacheRequestOutcomeListener**: 處理緩存請求結果的介面和類別。
   - **CacheRequestOutcome**: 處理緩存請求結果的介面和類別。

5. **Life Cycle and Utility**
   - **TerminationNotificationListener**: 這些用於處理組件終止的通知和事件。
   - **TerminationEvent**: 這些用於處理組件終止的通知和事件。

### SolClient

這個類封裝了使用 Solace 消息傳遞系統的一系列功能，主要用於創建連接、管理訂閱、發送和接收消息等。

- **`__init__`**

  - **目的**：使用必要的配置參數和用戶憑證初始化 `SolClient`。
  - **參數**：`clientname`、`sol_config`、`username`（可選）。
  - **操作**：設置連接狀態的默認值、消息服務對象以及用戶憑證。

- **`create_connection`**

  - **目的**：使用提供的配置和處理程序與 Solace 代理建立連接。
  - **參數**：`message_handler` 和 `service_handler`。
  - **操作**：
    - 構建基於會話參數的代理屬性。
    - 初始化帶有重連策略和其他配置的 `MessagingService`。
    - 連接服務並配置消息接收器和發布器。
    - 啟動消息接收器和發布器。
    - 檢查連接狀態並返回適當的響應代碼。

- **`disconnect`**

  - **目的**：正確斷開消息服務並清理資源。
  - **操作**：
    - 終止消息接收器。
    - 斷開消息服務的連接。
    - 重置連接狀態標誌。

- **`send_request`**

  - **目的**：向指定主題發送請求消息並等待回應。
  - **參數**：`topic`、`msg`（消息內容）。
  - **操作**：
    - 構建消息並將其發送到指定主題。
    - 在指定超時內等待並處理回覆。
    - 處理發送消息過程中的異常和錯誤。

- **`add_subscription`**

  - **目的**：訂閱指定主題以接收其上發布的消息。
  - **參數**：`topic`。
  - **操作**：向消息接收器添加主題訂閱並進行錯誤處理。

- **`remove_subscription`**

  - **目的**：從主題中移除現有的訂閱。
  - **參數**：`topic`。
  - **操作**：從消息接收器中移除主題訂閱並進行錯誤處理。

- **`request_cached_only`**

  - **目的**：僅請求來自緩存的消息。
  - **參數**：`cachename`、`topic`、`timeout`、`cacheId`。
  - **操作**：
    - 創建僅緩存的訂閱請求。
    - 提交緩存請求並使用自定義監聽器監聽完成或失敗。

- **`GetRequestID`**

  - **目的**：為需要唯一識別的操作生成唯一的請求 ID。
  - **操作**：生成基於時間戳的 ID，確保它大於上次使用的 ID。

### SorApi

該類別是一個封裝了與證券訂單路由系統（SORS）溝通的介面。

這個類別提供了一系列的功能，允許用戶連接、發送請求、接收和處理回報等，也就是說，這個是連接券商下單系統的本體 API。

由於這個 API 內大部分的功能被封裝在 .dll 檔案內，所以也是元富證券為何要限定使用 Windows 系統的原因。

本來我們想把這些 .dll 檔案也一起拆了，但是拆了也不能做什麼，所以就先算了。

總之，在其他程式中，若有看到類似 `OnSorConnect` 這種，有 `Sor` 的字眼，那就是會送到這個模組來使用功能。

### MarketTrader

這是一個 ABC 類別，定義了交易的基本功能，但沒有任何實作，需要我們自己完成。

定義內容如下：

```python title="MasterTradePy\64bit\MasterTradePy\model.py"
class MarketTrader(metaclass=ABCMeta):
    @ abstractmethod
    def OnNewOrderReply(self, data) -> None:
        pass

    @ abstractmethod
    def OnChangeReply(self, data) -> None:
        pass

    @ abstractmethod
    def OnCancelReply(self, data) -> None:
        pass

    @ abstractmethod
    def OnReport(self, data) -> None:
        pass

    @ abstractmethod
    def OnAnnouncementEvent(self, data)->None:
        pass

    @ abstractmethod
    def OnSystemEvent(self, event: SystemEvent) -> None:
        pass

    @ abstractmethod
    def OnError(self, error: MTPYError):
        pass

    @ abstractmethod
    def OnReqResult(self, workid: str, data):
        pass
```

### Order

這是一個訂單的 `dataclass` 類別，定義了訂單內所有資料。

```python title="MasterTradePy\64bit\MasterTradePy\model.py"
@dataclass
class Order:
    sorRID: str = field(init = False, default='')
    exchange: Exchange = field(init = False, default=Exchange.TWSE)
    tradingSession: TradingSession = ""
    side: Side = field(default="")
    symbol: str = field(default="")
    priceType: PriceType = field(default="")
    price: str = field(default="")
    tradingUnit: TradingUnit = field(default=0)
    qty: str = field(default="")
    orderType: OrderType = field(default="")
    tradingType: TradingType = field(init = False, default="")
    brokerNo: str = field(init = False, default="")
    userDef: str = ""
    tradingAccount: str = ""
    ordNo: str = field(init = False, default="")
    trxTime: str = field(init = False, default="")
    lastdealTime: str = field(init = False, default="")
    status: str = field(init = False, default="")
    leavesQty: str = field(init = False, default="")
    cumQty: str = field(init = False, default="")
    dealPri: str = field(init = False, default="")
    tableName : str = ""
```

### ReportOrder

:::warning
我們不推薦任何股票，所有關於證券戶內的股票資訊都會以馬賽克處理。
:::

這是呼叫 API 的時候實際上會拿到的訂單回報資料。

```python title="MasterTradePy\64bit\MasterTradePy\model.py"
@dataclass
class ReportOrder:
    orgOrder: Order
    order: Order
    lastMessage: str = field(init = False, default="")
    scBalance: str =  field(init = False, default="")
```

它包含了 `orgOrder` 和 `order` 兩種狀態，我們認為這可能代表原始的訂單和經過交易所確認後的訂單。

以下實際展示從系統中取得之委託回報與成交回報內容：

- **1. 委託回報**

  ```python
  ReportOrder(
    orgOrder=Order(
      sorRID='592zj00420005coSG200',
      exchange=<Exchange.TWSE: 'TWSE'>,
      tradingSession='N',
      side='S',
      symbol='2002',
      priceType='L',
      price='24',
      tradingUnit=0,
      qty='2000',
      orderType='R',
      tradingType='',
      brokerNo='',
      userDef='',
      tradingAccount='我是馬賽克',
      ordNo='',
      trxTime='',
      lastdealTime='',
      status='',
      leavesQty='',
      cumQty='',
      dealPri='',
      tableName=''
    ),
    order=Order(
      sorRID='',
      exchange=<Exchange.TWSE: 'TWSE'>,
      tradingSession='N',
      side='S',
      symbol='2002',
      priceType='L',
      price='24',
      tradingUnit=0,
      qty='2000',
      orderType='R',
      tradingType='',
      brokerNo='',
      userDef='',
      tradingAccount='我是馬賽克',
      ordNo='j0042',
      trxTime='08:31:31.926000',
      lastdealTime='',
      status='101)委託已接受(交易所已接受)',
      leavesQty='2000',
      cumQty='0',
      dealPri='',
      tableName='ORD:TwsOrd'),
      lastMessage='',
      scBalance=''
    )
  )
  ```

- **2. 成交回報**

  ```python
  ReportOrder(
    orgOrder=Order(
      sorRID='592zj00390005coPW400',
      exchange=<Exchange.TWSE: 'TWSE'>,
      tradingSession='N',
      side='S',
      symbol='3481',
      priceType='L',
      price='13.7',
      tradingUnit=0,
      qty='4000',
      orderType='R',
      tradingType='',
      brokerNo='',
      userDef='',
      tradingAccount='我是馬賽克',
      ordNo='',
      trxTime='',
      lastdealTime='',
      status='',
      leavesQty='',
      cumQty='',
      dealPri='',
      tableName=''
    ),
    order=Order(
      sorRID='',
      exchange=<Exchange.TWSE: 'TWSE'>,
      tradingSession='N',
      side='S',
      symbol='3481',
      priceType='L',
      price='13.7',
      tradingUnit=0,
      qty='4000',
      orderType='R',
      tradingType='',
      brokerNo='',
      userDef='',
      tradingAccount='我是馬賽克',
      ordNo='j0039',
      trxTime='08:31:31.926000',
      lastdealTime='09:00:11.609000',
      status='111)全部成交',
      leavesQty='0',
      cumQty='4000',
      dealPri='',
      tableName='ORD:TwsOrd'),
      lastMessage='',
      scBalance=''
    )
  )
  ```

:::warning
在我們的測試中，訂單資料的 `ReportOrder.order.dealPri` 欄位即使在成交之後是仍是空的，所以我們無法得知成交價格。

姑且懷疑是一個 bug，但我們無法確定。
:::

### MasterTradeAPI

這個是在最外層，也就是讓我們使用者直接用來操作下單的 API。

在這裡，我們跟著元富證券提供的範例程式: [**sample.py**](https://github.com/DocsaidLab/AutoTraderX/blob/main/MasterLink_PythonAPI/MasterTradePy/MasterTradePy/64bit/sample.py)，初步看一下使用方式：

- **Step 1: 創建一個 `MarketTrader` 類別實例，並作為輸入參數**

  ```python
  api = MasterTradeAPI(MarketTrader())
  ```

- **Step 2: 連接到元富證券的下單系統**

  ```python
  # 連接目標是固定的
  api.SetConnectionHost('solace140.masterlink.com.tw:55555')

  # 輸入帳號密碼，並指定登入狀態
  # is_sim: 是否連線模擬主機，True 為連接模擬主機，False 為連接正式主機
  # is_force: 是否強制登入，True 為強制登入，False 為正常登入
  # is_event: 是否為特定事件（如競賽）的登入，True 為啟用事件，False 為不啟用事件
  rc = api.Login(username, password, is_sim, is_force, is_event)
  ```

  在登入的過程中，程式使用 `SolClient` 來建立系統訊息的連接；使用 `SorApi` 來建立下單系統的連接。

  登入後會進行雙因子驗證：

  ```python
  accounts = [x[4:] for x in api.accounts]
  rcc = api.CheckAccs(tradingAccounts=accounts)
  ```

  從這個雙因子驗證採用 List 的方式中，我們可以看出這個 API 是支援多帳號的。

  也就是說，我們可以重複呼叫 `api.Login` 來登入不同的帳號，然後在交易時指定使用哪一個帳號來下單。

- **Step 3: 下單**

  下單時，必須要完成一個 `Order` 的物件設定，然後呼叫 `api.NewOrder` 來下單。

  ```python
  from MasterTradePy.model import *

  symbol = input(u'請輸入欲買進股票代號:')
  api.ReqBasic(symbol)
  account = input(u'請輸入下單帳號:')
  price = input(u'請輸入欲買進股票價格(空白表示市價下單):')
  qty = input(u'請輸入欲買進股票股數(1張請輸入1000):')
  orderTypeSymbol = input(u'請輸入類別(I:IOC, F:FOK, 其他:ROD):')

  orderType = OrderType.ROD
  if orderTypeSymbol == 'I':
      orderType = OrderType.IOC
  elif orderTypeSymbol == 'F':
      orderType = OrderType.FOK

  if not price:
      priceType = PriceType.MKT
  else:
      priceType = PriceType.LMT

  order = Order(tradingSession=TradingSession.NORMAL,
              side=Side.Buy,
              symbol=symbol,
              priceType=priceType,
              price=price,
              tradingUnit=TradingUnit.COMMON,
              qty=qty,
              orderType=orderType,
              tradingAccount=account,
              userDef='')
  rcode = api.NewOrder(order)
  if rcode == RCode.OK:
      print(u'已送出委託')
  else:
      print(u'下單失敗! 請再次執行程式，依據回報資料修正輸入')
  ```

- **Step 4: 改價**

  先調用 `OrderPriceChange` 來設定單號，再使用 `api.ChangeOrderPrice` 來改價。

  ```python
  from MasterTradePy.model import *

  ordNo = input(u'請輸入單號:')
  account = input(u'請輸入下單帳號:')
  price = input(u'請輸入股票價格(空白表示市價下單):')
  replaceOrder = OrderPriceChange(ordNo=ordNo, price=price,tradingAccount=account)

  rcode = api.ChangeOrderPrice(replaceOrder)
  if rcode == RCode.OK:
      print(u'已送出委託')
  else:
      print(u'改價失敗! 請再次執行程式，依據回報資料修正輸入')
  ```

- **Step 5: 改量**

  先調用 `OrderQtyChange` 來設定單號，再使用 `api.ChangeOrderQty` 來改價。

  ```python
  from MasterTradePy.model import *

  ordNo = input(u'請輸入單號:')
  account = input(u'請輸入下單帳號:')
  qty = input(u'請輸入股票股數(1張請輸入1000):')
  replaceOrder = OrderQtyChange(ordNo=ordNo, qty=qty, tradingAccount=account)

  rcode = api.ChangeOrderQty(replaceOrder)
  if rcode == RCode.OK:
      print(u'已送出委託')
  else:
      print(u'改量失敗! 請再次執行程式，依據回報資料修正輸入')
  ```

- **Step 6: 其他功能**

  - **`api.QryRepAll`**: 查詢所有委託
  - **`api.QryRepDeal`**: 查詢成交回報
  - **`api.ReqInventoryOpen`**: 查詢期初庫存
  - **`api.ReqInventoryRayinTotal`**: 查詢庫存
  - **`api.QrySecInvQty_Rayin`**: 查詢或有券源
  - **`api.QryProdCrQty_Rayin`**: 查詢資券配額
