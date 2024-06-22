---
sidebar_position: 4
---

# Order System Analysis

## MasterTradePy

After downloading the Python module from MasterLink, install the order system using the following command:

```powershell
pip install .\MasterLink_PythonAPI\MasterTradePy\MasterTradePy\64bit\MasterTradePy-0.0.23-py3-none-win_amd64.whl
```

:::tip
At the time of use, the package version was `0.0.23`.
:::

## Official Technical Documentation

- [**MasterLink - Order API**](https://mlapi.masterlink.com.tw/web_api/service/document/python-trade)
- [**Official Code Sample: sample.py**](https://github.com/DocsaidLab/AutoTraderX/blob/main/MasterLink_PythonAPI/MasterTradePy/MasterTradePy/64bit/sample.py)

## Core Modules

We have broken down the MasterLink Python module into the following core modules:

### Solace

:::tip
Solace is a company providing high-performance messaging middleware and related services, particularly suitable for real-time data streams and event-driven architectures.
:::

You will see many `Solace` classes in the code, which are used to communicate with the Solace message broker.

1. **Messaging Service**

   - **MessagingService**: The main interface for messaging services, used to configure and establish connections with the Solace message broker.
   - **ReconnectionListener**: Interfaces for handling connection interruptions and reconnections.
   - **ReconnectionAttemptListener**: Interfaces for handling connection interruptions and reconnections.
   - **ServiceInterruptionListener**: Interfaces for handling connection interruptions and reconnections.
   - **RetryStrategy**: Interface for defining message retransmission strategies.
   - **ServiceEvent**: Handles events related to service lifecycle.

2. **Topic Subscription**

   - **TopicSubscription**: Used to subscribe to messages for specific topics.

3. **Message Receiver and Publisher**

   - **MessageHandler**: Used to handle received messages.
   - **InboundMessage**: Used to handle received messages.
   - **DirectMessageReceiver**: Receivers that directly receive messages from the broker.
   - **OutboundMessage**: Represents messages to be sent.
   - **RequestReplyMessagePublisher**: Message publishers used for publishing requests and receiving replies.
   - **PublishFailureListener**: Handles events related to message publication failures.
   - **FailedPublishEvent**: Handles events related to message publication failures.

4. **Topic and Caching**

   - **Topic**: Represents message topics, which can be dynamic or fixed.
   - **CachedMessageSubscriptionRequest**: Handles subscription requests for cached messages.
   - **CacheRequestOutcomeListener**: Interfaces and classes for handling cache request outcomes.
   - **CacheRequestOutcome**: Interfaces and classes for handling cache request outcomes.

5. **Life Cycle and Utility**
   - **TerminationNotificationListener**: Handles notifications and events related to component termination.
   - **TerminationEvent**: Handles notifications and events related to component termination.

### SolClient

This class encapsulates a series of functionalities for using the Solace messaging system, mainly for creating connections, managing subscriptions, sending, and receiving messages.

- **`__init__`**

  - **Purpose**: Initialize `SolClient` with necessary configuration parameters and user credentials.
  - **Parameters**: `clientname`, `sol_config`, `username` (optional).
  - **Operation**: Sets default values for connection state, messaging service object, and user credentials.

- **`create_connection`**

  - **Purpose**: Establish a connection with the Solace broker using the provided configuration and handlers.
  - **Parameters**: `message_handler` and `service_handler`.
  - **Operation**:
    - Constructs broker properties based on session parameters.
    - Initializes `MessagingService` with reconnection strategy and other configurations.
    - Connects the service and configures message receivers and publishers.
    - Starts message receivers and publishers.
    - Checks connection status and returns appropriate response codes.

- **`disconnect`**

  - **Purpose**: Properly disconnect the messaging service and clean up resources.
  - **Operation**:
    - Terminates message receivers.
    - Disconnects the messaging service.
    - Resets connection state flags.

- **`send_request`**

  - **Purpose**: Sends a request message to the specified topic and waits for a reply.
  - **Parameters**: `topic`, `msg` (message content).
  - **Operation**:
    - Constructs the message and sends it to the specified topic.
    - Waits for and processes the reply within a specified timeout.
    - Handles exceptions and errors during the message-sending process.

- **`add_subscription`**

  - **Purpose**: Subscribes to the specified topic to receive messages published on it.
  - **Parameters**: `topic`.
  - **Operation**: Adds topic subscription to the message receiver and handles errors.

- **`remove_subscription`**

  - **Purpose**: Removes an existing subscription from the topic.
  - **Parameters**: `topic`.
  - **Operation**: Removes topic subscription from the message receiver and handles errors.

- **`request_cached_only`**

  - **Purpose**: Requests messages from the cache only.
  - **Parameters**: `cachename`, `topic`, `timeout`, `cacheId`.
  - **Operation**:
    - Creates a cache-only subscription request.
    - Submits the cache request and listens for completion or failure with custom listeners.

- **`GetRequestID`**

  - **Purpose**: Generates a unique request ID for operations requiring unique identification.
  - **Operation**: Generates a timestamp-based ID, ensuring it is greater than the previously used ID.

### SorApi

This class is an interface encapsulating communication with the Securities Order Routing System (SORS).

It provides a series of functions allowing users to connect, send requests, receive, and process responses. This is the core API for connecting to the broker's order system.

Most of its functionality is encapsulated in `.dll` files, which is why MasterLink restricts usage to Windows systems.

We initially considered decompiling these `.dll` files, but it wouldn't provide much benefit, so we decided against it.

In other programs, any reference to `Sor` such as `OnSorConnect` indicates the use of this module's functionality.

### MarketTrader

This is an abstract base class (ABC) defining the basic functions of trading but without any implementation, requiring us to complete it ourselves.

The defined content is as follows:

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

This is a `dataclass` defining all the data within an order.

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
We do not recommend any stocks; all stock information in the securities account will be blurred.
:::

This is the order report data that you receive when calling the API.

```python title="MasterTradePy\64bit\MasterTradePy\model.py"
@dataclass
class ReportOrder:
    orgOrder: Order
    order: Order
    lastMessage: str = field(init = False, default="")
    scBalance: str =  field(init = False, default="")
```

It contains `orgOrder` and `order` states, which we believe represent the original order and the order confirmed by the exchange.

Below are actual examples of order and trade reports retrieved from the system:

- **1. Order Report**

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
      tradingAccount='blurred',
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
      tradingAccount='blurred',
      ordNo='j0042',
      trxTime='08:31:31.926000',
      lastdealTime='',
      status='101)Order Accepted (Exchange Accepted)',
      leavesQty='2000',
      cumQty='0',
      dealPri='',
      tableName='ORD:TwsOrd'),
      lastMessage='',
      scBalance=''
      )
    )

  ```

- **2. Trade Report**

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
      tradingAccount='blurred',
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
      tradingAccount='blurred',
      ordNo='j0039',
      trxTime='08:31:31.926000',
      lastdealTime='09:00:11.609000',
      status='111)Fully Filled',
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
In our tests, the `ReportOrder.order.dealPri` field remained empty even after the trade was executed, so we couldn't determine the execution price.

We suspect this is a bug but cannot confirm it.
:::

### MasterTradeAPI

This is the top-level API used by users to operate the order system directly.

Following the example provided by MasterLink: [**sample.py**](https://github.com/DocsaidLab/AutoTraderX/blob/main/MasterLink_PythonAPI/MasterTradePy/MasterTradePy/64bit/sample.py), let's take a preliminary look at how to use it:

- **Step 1: Create an instance of the `MarketTrader` class and use it as an input parameter**

  ```python
  api = MasterTradeAPI(MarketTrader())
  ```

- **Step 2: Connect to the MasterLink order system**

  ```python
  # The target connection is fixed
  api.SetConnectionHost('solace140.masterlink.com.tw:55555')

  # Enter the account password and specify the login status
  # is_sim: Whether to connect to the simulation host, True for connecting to the simulation host, False for connecting to the official host
  # is_force: Whether to force login, True for forced login, False for normal login
  # is_event: Whether to enable event-specific login, True for enabling events, False for not enabling events
  rc = api.Login(username, password, is_sim, is_force, is_event)
  ```

  During the login process, the program uses `SolClient` to establish a system message connection and `SorApi` to establish an order system connection.

  After logging in, two-factor authentication will take place:

  ```python
  accounts = [x[4:] for x in api.accounts]
  rcc = api.CheckAccs(tradingAccounts=accounts)
  ```

  The two-factor authentication using a list format indicates that this API supports multiple accounts.

  This means you can repeatedly call `api.Login` to log into different accounts and specify which account to use for trading.

- **Step 3: Placing an Order**

  To place an order, you must create an `Order` object and call `api.NewOrder` to place the order.

  ```python
  from MasterTradePy.model import *

  symbol = input('Enter the stock symbol to buy:')
  api.ReqBasic(symbol)
  account = input('Enter the trading account:')
  price = input('Enter the stock price to buy (leave blank for market order):')
  qty = input('Enter the quantity of stock to buy (enter 1000 for 1 lot):')
  orderTypeSymbol = input('Enter the order type (I: IOC, F: FOK, others: ROD):')

  orderType = OrderType.ROD
  if orderTypeSymbol == 'I':
      orderType = OrderType.IOC
  elif orderTypeSymbol == 'F':
      orderType = OrderType.FOK

  if not price:
      priceType = PriceType.MKT
  else:
      priceType = PriceType.LMT

  order = Order(
              tradingSession=TradingSession.NORMAL,
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
      print('Order placed successfully')
  else:
      print('Order placement failed! Please run the program again and correct the input based on the feedback')
  ```

- **Step 4: Changing the Order Price**

  First, use `OrderPriceChange` to set the order number, then use `api.ChangeOrderPrice` to change the price.

  ```python
  from MasterTradePy.model import *

  ordNo = input('Enter the order number:')
  account = input('Enter the trading account:')
  price = input('Enter the stock price (leave blank for market order):')
  replaceOrder = OrderPriceChange(ordNo=ordNo, price=price, tradingAccount=account)

  rcode = api.ChangeOrderPrice(replaceOrder)
  if rcode == RCode.OK:
      print('Order price changed successfully')
  else:
      print('Order price change failed! Please run the program again and correct the input based on the feedback')
  ```

- **Step 5: Changing the Order Quantity**

  First, use `OrderQtyChange` to set the order number, then use `api.ChangeOrderQty` to change the quantity.

  ```python
  from MasterTradePy.model import *

  ordNo = input('Enter the order number:')
  account = input('Enter the trading account:')
  qty = input('Enter the stock quantity (enter 1000 for 1 lot):')
  replaceOrder = OrderQtyChange(ordNo=ordNo, qty=qty, tradingAccount=account)

  rcode = api.ChangeOrderQty(replaceOrder)
  if rcode == RCode.OK:
      print('Order quantity changed successfully')
  else:
      print('Order quantity change failed! Please run the program again and correct the input based on the feedback')
  ```

- **Step 6: Other Functions**

  - **`api.QryRepAll`**: Query all orders
  - **`api.QryRepDeal`**: Query trade reports
  - **`api.ReqInventoryOpen`**: Query initial inventory
  - **`api.ReqInventoryRayinTotal`**: Query inventory
  - **`api.QrySecInvQty_Rayin`**: Query available stock sources
  - **`api.QryProdCrQty_Rayin`**: Query collateral allocations
