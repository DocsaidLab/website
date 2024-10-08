---
sidebar_position: 3
---

# 報價系統分析

## SolPYAPI

當你下載完元富證券的 Python 模組後，使用以下指令安裝報價系統：

```powershell
pip install .\MasterLink_PythonAPI\SolPYAPI\PY_TradeD-0.1.15-py3-none-any.whl
```

:::tip
在我們使用時，該套件的版本為 `0.1.15`。
:::

## 官方技術文件

- [**元富證券-報價 API**](https://mlapi.masterlink.com.tw/web_api/service/document/python-quote)
- [**官方程式範例：Sample_D.py**](https://github.com/DocsaidLab/AutoTraderX/blob/main/MasterLink_PythonAPI/SolPYAPI/Sample_D.py)

## 核心模組

我們把元富證券的 Python 模組拆解成以下幾個核心模組：

### ProductBasic

這是用於記錄和回傳股票相關資訊的類別。

<details>
  <summary>點選展開物件屬性</summary>

    | No. | 欄位名稱                        | 資料類型 | 格式     | 說明                                                                                                                                      |
    | --- | ------------------------------- | -------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
    | 1   | Exchange                        | str      |          | 交易所(TWSE、TAIFEX)                                                                                                                      |
    | 2   | Symbol                          | str      |          | 商品代號(TWSE、TAIFEX)                                                                                                                    |
    | 3   | Category                        | str      |          | 商品分類(TWSE、TAIFEX)                                                                                                                    |
    | 4   | TodayRefPrice                   | str      |          | 參考價(TAIFEX)                                                                                                                            |
    | 5   | RiseStopPrice                   | str      |          | 漲停價(TWSE、TAIFEX)                                                                                                                      |
    | 6   | FallStopPrice                   | str      |          | 跌停價(TWSE、TAIFEX)                                                                                                                      |
    | 7   | ChineseName                     | str      | UTF-8    | 商品中文名稱(TWSE)                                                                                                                        |
    | 8   | PreTotalMatchQty                | str      |          | 上一交易日成交總量(TWSE、TAIFEX)                                                                                                          |
    | 9   | PreTodayRefPrice                | str      |          | 上一交易日參考價(TWSE、TAIFEX)                                                                                                            |
    | 10  | PreClosePrice                   | str      |          | 上一交易日收盤價(TWSE、TAIFEX)                                                                                                            |
    | 11  | IndustryCategory                | str      |          | 參考"產業別代碼表" 產業別(TWSE)                                                                                                           |
    | 12  | StockCategory                   | str      |          | 參考“證券別代碼表” 證券別(TWSE)                                                                                                           |
    | 13  | BoardRemark                     | str      |          | 板別註記(TWSE)                                                                                                                            |
    | 14  | ClassRemark                     | str      |          | 類股註記(TWSE)                                                                                                                            |
    | 15  | StockAnomalyCode                | str      |          | 參考"股票異常代碼表" 股票異常代碼(TWSE)                                                                                                   |
    | 16  | NonTenParValueRemark            | str      |          | 非 10 元面額註記(TWSE)                                                                                                                    |
    | 17  | AbnormalRecommendationIndicator | str      |          | 異常推介個股註記(TWSE)                                                                                                                    |
    | 18  | AbnormalSecuritiesIndicator     | str      |          | 異常推介個股註記(TWSE)                                                                                                                    |
    | 19  | DayTradingRemark                | str      |          | "0"：預設值 "A"：可先買後賣或先賣後買現股當沖證券 "B"：時表示為 可先買後賣現股當沖證券 SPACE：表示為不可現股當沖證券 可現股當沖註記(TWSE) |
    | 20  | TradingUnit                     | str      |          | 交易單位(TWSE)                                                                                                                            |
    | 21  | TickSize                        | str      |          | 最小跳動單位(TWSE)                                                                                                                        |
    | 22  | prodKind                        | str      |          | 契約種類(TAIFEX)                                                                                                                          |
    | 23  | strikePriceDecimalLocator       | str      |          | 選擇權商品代號之履約價小數位數(TAIFEX)                                                                                                    |
    | 24  | PreTotalTradingAmount           | str      |          | 上一交易日成交總額(TWSE)                                                                                                                  |
    | 25  | DecimalLocator                  | str      |          | 價格小數位數(TAIFEX)                                                                                                                      |
    | 26  | BeginDate                       | str      | YYYYMMDD | 上市日期(TAIFEX)                                                                                                                          |
    | 27  | EndDate                         | str      | YYYYMMDD | 下市日期(TAIFEX)                                                                                                                          |
    | 28  | FlowGroup                       | str      |          | 流程群組(TAIFEX)                                                                                                                          |
    | 29  | DeliveryDate                    | str      | YYYYMMDD | 最後結算日(TAIFEX)                                                                                                                        |
    | 30  | DynamicBanding                  | str      |          | Y:適用, N:不適用 適用動態價格穩定(TAIFEX)                                                                                                 |
    | 31  | ContractSymbol                  | str      |          | 契約代號(TAIFEX)                                                                                                                          |
    | 32  | ContractName                    | str      |          | 契約中文名稱(TAIFEX)                                                                                                                      |
    | 33  | StockID                         | str      |          | 現貨股票代碼(TAIFEX)                                                                                                                      |
    | 34  | StatusCode                      | str      |          | N：正常 P：暫停交易 U：即將上市 狀態碼(TAIFEX)                                                                                            |
    | 35  | Currency                        | str      |          | 幣別(TAIFEX)                                                                                                                              |
    | 36  | AcceptQuoteFlag                 | str      |          | 是否可報價(TAIFEX)                                                                                                                        |
    | 37  | BlockTradeFlag                  | str      |          | Y:可 N:不可 是否可鉅額交易(TAIFEX)                                                                                                        |
    | 38  | ExpiryType                      | str      |          | S:標準 W:週 到期別(TAIFEX)                                                                                                                |
    | 39  | UnderlyingType                  | str      |          | E S:個股 現貨類別(TAIFEX)                                                                                                                 |
    | 40  | MarketCloseGroup                | str      |          | 參考"商品收盤時間群組表" 商品收盤時間群組(TAIFEX)                                                                                         |
    | 41  | EndSession                      | str      |          | 一般交易時段：0 盤後交易時段：1 交易時段(TAIFEX)                                                                                          |
    | 42  | isAfterHours                    | str      |          | 早盤 : 0 午盤: 1 早午盤辨識(TAIFEX)                                                                                                       |

</details>

### ProductTick

即時交易明細資訊。

<details>
    <summary>點選展開物件屬性</summary>

      | No.  | 欄位名稱                   | 資料類型  | 格式          | 說明                                                                                      |
      |------|----------------------------|-----------|---------------|-------------------------------------------------------------------------------------------|
      | 1    | Exchange                   | str       |               | 交易所(TWSE、TAIFEX)                                                                      |
      | 2    | Symbol                     | str       |               | 商品代號(TWSE、TAIFEX)                                                                    |
      | 3    | MatchTime                  | str       | %H:%M:%S.%f   | 成交資料時間(交易所) (TWSE、TAIFEX)                                                       |
      | 4    | OrderBookTime              | str       | %H:%M:%S.%f   | 五檔資料時間(交易所) (TWSE、TAIFEX)                                                       |
      | 5    | TxSeq                      | str       |               | 交易所序號(成交資訊) (TWSE、TAIFEX)                                                       |
      | 6    | ObSeq                      | str       |               | 交易所序號(五檔資訊) (TWSE、TAIFEX)                                                       |
      | 7    | IsTxTrail                  | bool      |               | 0: 非試撮，1: 試撮 是否為成交試撮資料(TWSE、TAIFEX)                                      |
      | 8    | Is5QTrial                  | bool      |               | 0: 非試撮，1: 試撮 是否為五檔試撮資料(TWSE、TAIFEX)                                      |
      | 9    | IsTrail                    | bool      |               | 0: 非試撮，1: 試撮 是否為試撮資料(TWSE、TAIFEX)                                          |
      | 10   | DecimalLocator             | str       |               | 價格欄位小數位數(TAIFEX)                                                                  |
      | 11   | MatchPrice                 | str       |               | 成交價(TWSE、TAIFEX)                                                                      |
      | 12   | MatchQty                   | str       |               | 商品成交量(TAIFEX)                                                                        |
      | 13   | MatchPriceList             | list      |               | 一筆行情, 多筆成交價(TWSE、TAIFEX)                                                        |
      | 14   | MatchQtyList               | list      |               | 一筆行情, 多筆成交量(TWSE、TAIFEX)                                                        |
      | 15   | MatchBuyCount              | str       |               | 累計買進成交筆數(TAIFEX)                                                                  |
      | 16   | MatchSellCount             | str       |               | 累計賣出成交筆數(TAIFEX)                                                                  |
      | 17   | TotalMatchQty              | str       |               | 商品成交總量(TWSE、TAIFEX)                                                                |
      | 18   | TotalTradingAmount         | str       |               | 商品成交總額(TWSE、TAIFEX)                                                                |
      | 19   | TradingUnit                | str       |               | 交易單位(TWSE、TAIFEX)                                                                    |
      | 20   | DayHigh                    | str       |               | 當日最高價(TWSE、TAIFEX)                                                                  |
      | 21   | DayLow                     | str       |               | 當日最低價(TWSE、TAIFEX)                                                                  |
      | 22   | RefPrice                   | str       |               | 參考價(TWSE)                                                                              |
      | 23   | BuyPrice                   | list      |               | 五檔報價(買價) (TWSE、TAIFEX)                                                              |
      | 24   | BuyQty                     | list      |               | 五檔報價(買量) (TWSE、TAIFEX)                                                              |
      | 25   | SellPrice                  | list      |               | 五檔報價(賣價) (TWSE、TAIFEX)                                                              |
      | 26   | SellQty                    | list      |               | 五檔報價(賣量) (TWSE、TAIFEX)                                                              |
      | 27   | AllMarketAmount            | str       |               | 整體市場成交總額(TWSE)                                                                    |
      | 28   | AllMarketVolume            | str       |               | 整體市場成交數量(TWSE)                                                                    |
      | 29   | AllMarketCnt               | str       |               | 整體市場成交筆數(TWSE)                                                                    |
      | 30   | AllMarketBuyCnt            | str       |               | 整體市場委託買進筆數(TWSE)                                                                |
      | 31   | AllMarketSellCnt           | str       |               | 整體市場委託賣出筆數(TWSE)                                                                |
      | 32   | AllMarketBuyQty            | str       |               | 整體市場委託買進數量(TWSE)                                                                |
      | 33   | AllMarketSellQty           | str       |               | 整體市場委託賣出數量(TWSE)                                                                |
      | 34   | IsFixedPriceTransaction    | str       |               | 是否為定盤交易(TWSE)                                                                      |
      | 35   | OpenPrice                  | str       |               | 開盤價(TWSE、TAIFEX)                                                                      |
      | 36   | FirstDerivedBuyPrice       | str       |               | 衍生委託單第一檔買進價格(TAIFEX)                                                          |
      | 37   | FirstDerivedBuyQty         | str       |               | 衍生委託單第一檔買進價格數量(TAIFEX)                                                      |
      | 38   | FirstDerivedSellPrice      | str       |               | 衍生委託單第一檔賣出價格數量(TAIFEX)                                                      |
      | 39   | FirstDerivedSellQty        | str       |               | 衍生委託單第一檔賣出價格數量(TAIFEX)                                                      |
      | 40   | TotalBuyOrder              | str       |               | 買進累計委託筆數(TAIFEX)                                                                  |
      | 41   | TotalBuyQty                | str       |               | 買進累計委託合約數(TAIFEX)                                                                |
      | 42   | TotalSellOrder             | str       |               | 賣出累計委託筆數(TAIFEX)                                                                  |
      | 43   | TotalSellQty               | str       |               | 賣出累計委託合約數(TAIFEX)                                                                |
      | 44   | ClosePrice                 | str       |               | 收盤價(TAIFEX)                                                                            |
      | 45   | SettlePrice                | str       |               | 結算價(TAIFEX)                                                                            |

</details>

### RCode

這是一個用來記錄報價系統回傳的錯誤代碼的類別。

<details>
  <summary>點選展開物件屬性</summary>
| 值    | 名稱                                 | 說明                                      |
|-------|--------------------------------------|-------------------------------------------|
| 0     | OK                                   | 成功                                      |
| 1     | SOLCLIENT_WOULD_BLOCK                | API 呼叫會阻塞，但請求非阻塞模式              |
| 2     | SOLCLIENT_IN_PROGRESS                | API 呼叫正在進行中（非阻塞模式）               |
| 3     | SOLCLIENT_NOT_READY                  | API 無法完成，因為對象未準備好（例如，會話未連接） |
| 4     | SOLCLIENT_EOS                        | 結構化容器上的下一次操作返回了流結束            |
| 5     | SOLCLIENT_NOT_FOUND                  | 在 MAP 中查找命名字段未找到                   |
| 6     | SOLCLIENT_NOEVENT                    | 上下文無事件可處理                           |
| 7     | SOLCLIENT_INCOMPLETE                 | API 呼叫完成了部分但不是所有請求的功能          |
| 8     | SOLCLIENT_ROLLBACK                   | 當交易已回滾時，Commit() 返回此值             |
| 9     | SOLCLIENT_EVENT                      | SolClient 會話事件                          |
| 10    | CLIENT_ALREADY_CONNECTED             | 連線已建立                                  |
| 11    | CLIENT_ALREADY_DISCONNECTED          | 連線已斷線                                  |
| 12    | ANNOUNCEMENT                         | 公告訊息                                    |
| -1    | FAIL                                 | 失敗                                      |
| -2    | CONNECTION_REFUSED                   | 拒絕連線                                    |
| -3    | CONNECTION_FAIL                      | 連線失敗                                    |
| -4    | ALREADY_EXISTS                       | 目標物件已存在                               |
| -5    | NOT_FOUND                            | 目標物件不存在                               |
| -6    | CLIENT_NOT_READY                     | 連線尚未準備好                               |
| -7    | USER_SUBSCRIPTION_LIMIT_EXCEEDED     | 超過訂閱數上限                               |
| -8    | USER_NOT_APPLIED                     | 尚未申請                                    |
| -9    | USER_NOT_VERIFIED                    | 尚未驗證                                    |
| -10   | USER_VERIFICATION_FAIL               | 驗證失敗                                    |
| -11   | SUBSCRIPTION_FAIL                    | 訂閱商品失敗                                 |
| -12   | RECOVERY_FAIL                        | 回補失敗                                    |
| -13   | DOWNLOAD_PRODUCT_FAIL                | 下載基本資料檔失敗                            |
| -14   | MESSAGE_HANDLER_FAIL                 | 訊息處理錯誤                                 |
| -15   | FUNCTION_SUBSCRIPTION_LIMIT_EXCEEDED | 功能訂閱數超過上限                             |
| -16   | USER_NOT_VERIFIED_TWSE               | 尚未驗證 TWSE                                |
| -17   | USER_NOT_VERIFIED_TAIFEX             | 尚未驗證 TAIFEX                              |
| -18   | USER_NOT_VERIFIED_TWSE_TAIFEX        | 尚未驗證 TWSE&TAIFEX                          |
| -9999 | UNKNOWN_ERROR                        | 未知錯誤                                    |

</details>

### MarketDataMart

這是報價系統中的一個起始點，類別本身只定義了幾個方法，這些方法會用來觸發事件，例如成交訊息、委託訊息等。

在後續的使用中，必須手動掛載對應的事件處理器。

| Event                                | Description            |
| ------------------------------------ | ---------------------- |
| **Fire_OnSystemEvent**               | 系統訊息通知           |
| **Fire_MarketDataMart_ConnectState** | 系統連線狀態           |
| **Fire_OnUpdateBasic**               | 商品基本資料更新       |
| **Fire_OnUpdateProductBasicList**    | 商品基本資料列表更新   |
| **Fire_OnUpdateLastSnapshot**        | 商品最新快照更新       |
| **Fire_OnMatch**                     | 商品成交訊息           |
| **Fire_OnOrderBook**                 | 商品委託訊息           |
| **Fire_OnUpdateTotalOrderQty**       | 商品委託量累計更新     |
| **Fire_OnUpdateOptionGreeks**        | 選擇權商品 Greeks 更新 |
| **Fire_OnUpdateOvsBasic**            | 海外商品基本資料更新   |
| **Fire_OnUpdateOvsMatch**            | 海外商品成交資料更新   |
| **Fire_OnUpdateOvsOrderBook**        | 海外商品五檔資料更新   |

### Sol_D

這個類別的輸入參數是一個 `MarketDataMart` 類別的實例，並直接將 `MarketDataMart` 實例送進 `MasterQuoteDAPI` 類別中，產生另外一個實例。

我們查看 `MasterQuoteDAPI` 的實作中，會接收到 `MarketDataMart` 實例後，再次把它送進 `SolAPI` 類別中，產生另外一個實例。

好吧，這些都是他們的實作方式，先不管了。

---

在 `Sol_D` 類別中，裡面有提供幾個使用者常用的方法：

- `Sol_D.Login`: 登入
- `Sol_D.DisConnect`: 登出
- `Sol_D.Subscribe`: 訂閱商品報價
- `Sol_D.UnSubscribe`: 取消訂閱商品報價

還有一些事件處理器，必須先定義外部函數之後，用他們指定的方法掛載：

- `Sol_D.Set_OnLogEvent`: 設定登入事件處理器
- `Sol_D.Set_OnInterruptEvent`: 設定系統事件處理器
- `Sol_D.Set_OnLoginResultEvent_DAPI`: 設定登入結果事件處理器
- `Sol_D.Set_OnAnnouncementEvent_DAPI`: 設定公告事件處理器
- `Sol_D.Set_OnVerifiedEvent_DAPI`: 設定驗證事件處理器
- `Sol_D.Set_OnSystemEvent_DAPI`: 設定系統事件處理器
- `Sol_D.Set_OnUpdateBasic_DAPI`: 設定商品基本資料更新事件處理器
- `Sol_D.Set_OnMatch_DAPI`: 設定商品成交訊息事件處理器
