---
sidebar_position: 3
---

# Quotation System Analysis

## SolPYAPI

After downloading the Python module from MasterLink, install the quotation system using the following command:

```powershell
pip install .\MasterLink_PythonAPI\SolPYAPI\PY_TradeD-0.1.15-py3-none-any.whl
```

:::tip
At the time of use, the package version was `0.1.15`.
:::

## Official Technical Documentation

- [**MasterLink - Quotation API**](https://mlapi.masterlink.com.tw/web_api/service/document/python-quote)
- [**Official Code Sample: Sample_D.py**](https://github.com/DocsaidLab/AutoTraderX/blob/main/MasterLink_PythonAPI/SolPYAPI/Sample_D.py)

## Core Modules

We have deconstructed the MasterLink Python module into the following core modules:

### ProductBasic

This class is used to record and return stock-related information.

<details>
  <summary>Click to expand object properties</summary>

    | No. | Field Name                     | Data Type | Format  | Description                                                                                                                            |
    | --- | ------------------------------ | --------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------- |
    | 1   | Exchange                       | str       |         | Exchange (TWSE, TAIFEX)                                                                                                                |
    | 2   | Symbol                         | str       |         | Product Code (TWSE, TAIFEX)                                                                                                            |
    | 3   | Category                       | str       |         | Product Category (TWSE, TAIFEX)                                                                                                        |
    | 4   | TodayRefPrice                  | str       |         | Reference Price (TAIFEX)                                                                                                               |
    | 5   | RiseStopPrice                  | str       |         | Upper Limit Price (TWSE, TAIFEX)                                                                                                       |
    | 6   | FallStopPrice                  | str       |         | Lower Limit Price (TWSE, TAIFEX)                                                                                                       |
    | 7   | ChineseName                    | str       | UTF-8   | Product Chinese Name (TWSE)                                                                                                            |
    | 8   | PreTotalMatchQty               | str       |         | Previous Trading Day Total Volume (TWSE, TAIFEX)                                                                                       |
    | 9   | PreTodayRefPrice               | str       |         | Previous Trading Day Reference Price (TWSE, TAIFEX)                                                                                    |
    | 10  | PreClosePrice                  | str       |         | Previous Trading Day Closing Price (TWSE, TAIFEX)                                                                                      |
    | 11  | IndustryCategory               | str       |         | Industry Category (TWSE)                                                                                                               |
    | 12  | StockCategory                  | str       |         | Stock Category (TWSE)                                                                                                                  |
    | 13  | BoardRemark                    | str       |         | Board Remark (TWSE)                                                                                                                    |
    | 14  | ClassRemark                    | str       |         | Class Remark (TWSE)                                                                                                                    |
    | 15  | StockAnomalyCode               | str       |         | Stock Anomaly Code (TWSE)                                                                                                              |
    | 16  | NonTenParValueRemark           | str       |         | Non-10 Par Value Remark (TWSE)                                                                                                         |
    | 17  | AbnormalRecommendationIndicator| str       |         | Abnormal Recommendation Indicator (TWSE)                                                                                               |
    | 18  | AbnormalSecuritiesIndicator    | str       |         | Abnormal Securities Indicator (TWSE)                                                                                                   |
    | 19  | DayTradingRemark               | str       |         | Day Trading Remark (TWSE)                                                                                                              |
    | 20  | TradingUnit                    | str       |         | Trading Unit (TWSE)                                                                                                                    |
    | 21  | TickSize                       | str       |         | Minimum Tick Size (TWSE)                                                                                                               |
    | 22  | prodKind                       | str       |         | Contract Type (TAIFEX)                                                                                                                 |
    | 23  | strikePriceDecimalLocator      | str       |         | Strike Price Decimal Locator (TAIFEX)                                                                                                  |
    | 24  | PreTotalTradingAmount          | str       |         | Previous Trading Day Total Amount (TWSE)                                                                                               |
    | 25  | DecimalLocator                 | str       |         | Price Decimal Locator (TAIFEX)                                                                                                         |
    | 26  | BeginDate                      | str       | YYYYMMDD | Listing Date (TAIFEX)                                                                                                                  |
    | 27  | EndDate                        | str       | YYYYMMDD | Delisting Date (TAIFEX)                                                                                                                |
    | 28  | FlowGroup                      | str       |         | Flow Group (TAIFEX)                                                                                                                    |
    | 29  | DeliveryDate                   | str       | YYYYMMDD | Last Settlement Date (TAIFEX)                                                                                                          |
    | 30  | DynamicBanding                 | str       |         | Y: Applicable, N: Not Applicable, Dynamic Price Stabilization (TAIFEX)                                                                 |
    | 31  | ContractSymbol                 | str       |         | Contract Symbol (TAIFEX)                                                                                                               |
    | 32  | ContractName                   | str       |         | Contract Chinese Name (TAIFEX)                                                                                                         |
    | 33  | StockID                        | str       |         | Underlying Stock Code (TAIFEX)                                                                                                         |
    | 34  | StatusCode                     | str       |         | N: Normal, P: Trading Halt, U: To be Listed, Status Code (TAIFEX)                                                                      |
    | 35  | Currency                       | str       |         | Currency (TAIFEX)                                                                                                                      |
    | 36  | AcceptQuoteFlag                | str       |         | Accept Quote Flag (TAIFEX)                                                                                                             |
    | 37  | BlockTradeFlag                 | str       |         | Y: Yes, N: No, Block Trade Flag (TAIFEX)                                                                                               |
    | 38  | ExpiryType                     | str       |         | S: Standard, W: Weekly, Expiry Type (TAIFEX)                                                                                           |
    | 39  | UnderlyingType                 | str       |         | E S: Underlying Type (TAIFEX)                                                                                                          |
    | 40  | MarketCloseGroup               | str       |         | Market Close Group (TAIFEX)                                                                                                            |
    | 41  | EndSession                     | str       |         | 0: Regular Trading Session, 1: After-Hours Trading Session (TAIFEX)                                                                    |
    | 42  | isAfterHours                   | str       |         | 0: Morning Session, 1: Afternoon Session (TAIFEX)                                                                                      |

</details>

### ProductTick

Real-time transaction details.

<details>
    <summary>Click to expand object properties</summary>

    | No.  | Field Name                    | Data Type | Format        | Description                                                                          |
    | ---- | ----------------------------- | ----------| --------------| -------------------------------------------------------------------------------------|
    | 1    | Exchange                      | str       |               | Exchange (TWSE, TAIFEX)                                                              |
    | 2    | Symbol                        | str       |               | Product Code (TWSE, TAIFEX)                                                          |
    | 3    | MatchTime                     | str       | %H:%M:%S.%f   | Transaction Data Time (TWSE, TAIFEX)                                                 |
    | 4    | OrderBookTime                 | str       | %H:%M:%S.%f   | Order Book Data Time (TWSE, TAIFEX)                                                  |
    | 5    | TxSeq                         | str       |               | Exchange Sequence Number (Transaction Data) (TWSE, TAIFEX)                           |
    | 6    | ObSeq                         | str       |               | Exchange Sequence Number (Order Book Data) (TWSE, TAIFEX)                            |
    | 7    | IsTxTrail                     | bool      |               | 0: Non-Trail, 1: Trail, Is Transaction Trail Data (TWSE, TAIFEX)                     |
    | 8    | Is5QTrial                     | bool      |               | 0: Non-Trail, 1: Trail, Is Order Book Trail Data (TWSE, TAIFEX)                      |
    | 9    | IsTrail                       | bool      |               | 0: Non-Trail, 1: Trail, Is Trail Data (TWSE, TAIFEX)                                 |
    | 10   | DecimalLocator                | str       |               | Decimal Locator for Price Field (TAIFEX)                                             |
    | 11   | MatchPrice                    | str       |               | Transaction Price (TWSE, TAIFEX)                                                     |
    | 12   | MatchQty                      | str       |               | Product Transaction Volume (TAIFEX)                                                  |
    | 13   | MatchPriceList                | list      |               | Multiple Transaction Prices for One Market Quote (TWSE, TAIFEX)                      |
    | 14   | MatchQtyList                  | list      |               | Multiple Transaction Volumes for One Market Quote (TWSE, TAIFEX)                     |
    | 15   | MatchBuyCount                 | str       |               | Accumulated Buy Transaction Count (TAIFEX)                                           |
    | 16   | MatchSellCount                | str       |               | Accumulated Sell Transaction Count (TAIFEX)                                          |
    | 17   | TotalMatchQty                 | str       |               | Total Product Transaction Volume (TWSE, TAIFEX)                                      |
    | 18   | TotalTradingAmount            | str       |               | Total Product Transaction Amount (TWSE, TAIFEX)                                      |
    | 19   | TradingUnit                   | str       |               | Trading Unit (TWSE, TAIFEX)                                                          |
    | 20   | DayHigh                       | str       |               | Day High Price (TWSE, TAIFEX)                                                        |
    | 21   | DayLow                        | str       |               | Day Low Price (TWSE, TAIFEX)                                                         |
    | 22   | RefPrice                      | str       |               | Reference Price (TWSE)                                                               |
    | 23   | BuyPrice                      | list      |               | Order Book Buy Price (TWSE, TAIFEX)                                                  |
    | 24   | BuyQty                        | list      |               | Order Book Buy Quantity (TWSE, TAIFEX)                                               |
    | 25   | SellPrice                     | list      |               | Order Book Sell Price (TWSE, TAIFEX)                                                 |
    | 26   | SellQty                       | list      |               | Order Book Sell Quantity (TWSE, TAIFEX)                                              |
    | 27   | AllMarketAmount               | str       |               | Total Market Transaction Amount (TWSE)                                               |
    | 28   | AllMarketVolume               | str       |               | Total Market Transaction Volume (TWSE)                                               |
    | 29   | AllMarketCnt                  | str       |               | Total Market Transaction Count (TWSE)                                                |
    | 30   | AllMarketBuyCnt               | str       |               | Total Market Buy Order Count (TWSE)                                                  |
    | 31   | AllMarketSellCnt              | str       |               | Total Market Sell Order Count (TWSE)                                                 |
    | 32   | AllMarketBuyQty               | str       |               | Total Market Buy Order Quantity (TWSE)                                               |
    | 33   | AllMarketSellQty              | str       |               | Total Market Sell Order Quantity (TWSE)                                              |
    | 34   | IsFixedPriceTransaction       | str       |               | Fixed Price Transaction (TWSE)                                                       |
    | 35   | OpenPrice                     | str       |               | Opening Price (TWSE, TAIFEX)                                                         |
    | 36   | FirstDerivedBuyPrice          | str       |               | First Derived Buy Price (TAIFEX)                                                     |
    | 37   | FirstDerivedBuyQty            | str       |               | First Derived Buy Quantity (TAIFEX)                                                  |
    | 38   | FirstDerivedSellPrice         | str       |               | First Derived Sell Price (TAIFEX)                                                    |
    | 39   | FirstDerivedSellQty           | str       |               | First Derived Sell Quantity (TAIFEX)                                                 |
    | 40   | TotalBuyOrder                 | str       |               | Accumulated Buy Order Count (TAIFEX)                                                 |
    | 41   | TotalBuyQty                   | str       |               | Accumulated Buy Order Quantity (TAIFEX)                                              |
    | 42   | TotalSellOrder                | str       |               | Accumulated Sell Order Count (TAIFEX)                                                |
    | 43   | TotalSellQty                  | str       |               | Accumulated Sell Order Quantity (TAIFEX)                                             |
    | 44   | ClosePrice                    | str       |               | Closing Price (TAIFEX)                                                               |
    | 45   | SettlePrice                   | str       |               | Settlement Price (TAIFEX)                                                            |

</details>

### RCode

This class is used to record the error codes returned by the quotation system.

<details>
  <summary>Click to expand object properties</summary>

    | Value | Name                                  | Description                                             |
    |-------|---------------------------------------|---------------------------------------------------------|
    | 0     | OK                                    | Success                                                 |
    | 1     | SOLCLIENT_WOULD_BLOCK                 | API call would block but request is non-blocking        |
    | 2     | SOLCLIENT_IN_PROGRESS                 | API call is in progress (non-blocking mode)             |
    | 3     | SOLCLIENT_NOT_READY                   | API cannot complete as the object is not ready          |
    | 4     | SOLCLIENT_EOS                         | Next operation on structured container returned EOS     |
    | 5     | SOLCLIENT_NOT_FOUND                   | Named field not found in MAP lookup                     |
    | 6     | SOLCLIENT_NOEVENT                     | No events to process in the context                     |
    | 7     | SOLCLIENT_INCOMPLETE                  | API call completed some but not all requested functions |
    | 8     | SOLCLIENT_ROLLBACK                    | Commit() returns this value when a transaction is rolled back |
    | 9     | SOLCLIENT_EVENT                       | SolClient session event                                 |
    | 10    | CLIENT_ALREADY_CONNECTED              | Connection already established                          |
    | 11    | CLIENT_ALREADY_DISCONNECTED           | Connection already disconnected                         |
    | 12    | ANNOUNCEMENT                          | Announcement message                                    |
    | -1    | FAIL                                  | Failure                                                 |
    | -2    | CONNECTION_REFUSED                    | Connection refused                                      |
    | -3    | CONNECTION_FAIL                       | Connection failed                                       |
    | -4    | ALREADY_EXISTS                        | Target object already exists                            |
    | -5    | NOT_FOUND                             | Target object not found                                 |
    | -6    | CLIENT_NOT_READY                      | Connection not ready                                    |
    | -7    | USER_SUBSCRIPTION_LIMIT_EXCEEDED      | Subscription limit exceeded                             |
    | -8    | USER_NOT_APPLIED                      | Not applied                                             |
    | -9    | USER_NOT_VERIFIED                     | Not verified                                            |
    | -10   | USER_VERIFICATION_FAIL                | Verification failed                                     |
    | -11   | SUBSCRIPTION_FAIL                     | Subscription failed                                     |
    | -12   | RECOVERY_FAIL                         | Recovery failed                                         |
    | -13   | DOWNLOAD_PRODUCT_FAIL                 | Failed to download basic data file                      |
    | -14   | MESSAGE_HANDLER_FAIL                  | Message handling error                                  |
    | -15   | FUNCTION_SUBSCRIPTION_LIMIT_EXCEEDED  | Function subscription limit exceeded                    |
    | -16   | USER_NOT_VERIFIED_TWSE                | Not verified TWSE                                       |
    | -17   | USER_NOT_VERIFIED_TAIFEX              | Not verified TAIFEX                                     |
    | -18   | USER_NOT_VERIFIED_TWSE_TAIFEX         | Not verified TWSE & TAIFEX                              |
    | -9999 | UNKNOWN_ERROR                         | Unknown error                                           |

</details>

### MarketDataMart

This is an entry point in the quotation system, defining a few methods that trigger events such as transaction messages and order messages.

To use these methods, you must manually attach the corresponding event handlers.

| Event                                | Description                              |
| ------------------------------------ | ---------------------------------------- |
| **Fire_OnSystemEvent**               | System event notification                |
| **Fire_MarketDataMart_ConnectState** | System connection status                 |
| **Fire_OnUpdateBasic**               | Product basic data update                |
| **Fire_OnUpdateProductBasicList**    | Product basic data list update           |
| **Fire_OnUpdateLastSnapshot**        | Latest product snapshot update           |
| **Fire_OnMatch**                     | Product transaction message              |
| **Fire_OnOrderBook**                 | Product order message                    |
| **Fire_OnUpdateTotalOrderQty**       | Product order quantity update            |
| **Fire_OnUpdateOptionGreeks**        | Option product Greeks update             |
| **Fire_OnUpdateOvsBasic**            | Overseas product basic data update       |
| **Fire_OnUpdateOvsMatch**            | Overseas product transaction data update |
| **Fire_OnUpdateOvsOrderBook**        | Overseas product order book update       |

### Sol_D

This class takes an instance of the `MarketDataMart` class as input and passes it to the `MasterQuoteDAPI` class, creating another instance.

In the `MasterQuoteDAPI` implementation, the `MarketDataMart` instance is passed to the `SolAPI` class, creating yet another instance.

Well, these are their implementation methods, so we'll leave it at that.

---

In the `Sol_D` class, several commonly used methods are provided:

- `Sol_D.Login`: Log in
- `Sol_D.DisConnect`: Log out
- `Sol_D.Subscribe`: Subscribe to product quotes
- `Sol_D.UnSubscribe`: Unsubscribe from product quotes

Additionally, there are several event handlers that must be defined externally and attached using their specified methods:

- `Sol_D.Set_OnLogEvent`: Set log event handler
- `Sol_D.Set_OnInterruptEvent`: Set system event handler
- `Sol_D.Set_OnLoginResultEvent_DAPI`: Set login result event handler
- `Sol_D.Set_OnAnnouncementEvent_DAPI`: Set announcement event handler
- `Sol_D.Set_OnVerifiedEvent_DAPI`: Set verification event handler
- `Sol_D.Set_OnSystemEvent_DAPI`: Set system event handler
- `Sol_D.Set_OnUpdateBasic_DAPI`: Set product basic data update event handler
- `Sol_D.Set_OnMatch_DAPI`: Set product transaction message event handler
