---
sidebar_position: 3
---

# 価格システム分析

## SolPYAPI

元富証券の Python モジュールをダウンロードした後、以下のコマンドで価格システムをインストールします：

```powershell
pip install .\MasterLink_PythonAPI\SolPYAPI\PY_TradeD-0.1.15-py3-none-any.whl
```

:::tip
使用した際のバージョンは `0.1.15` です。
:::

## 公式技術文書

- [**元富証券-価格 API**](https://mlapi.masterlink.com.tw/web_api/service/document/python-quote)
- [**公式プログラム例：Sample_D.py**](https://github.com/DocsaidLab/AutoTraderX/blob/main/MasterLink_PythonAPI/SolPYAPI/Sample_D.py)

## コアモジュール

元富証券の Python モジュールを以下のコアモジュールに分けて解析しました：

### ProductBasic

これは株式に関連する情報を記録し、返すためのクラスです。

<details>
  <summary>オブジェクトの属性を展開</summary>

    | No. | フィールド名                        | データ型 | フォーマット | 説明                                                                                                                                      |
    | --- | ----------------------------------- | -------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
    | 1   | Exchange                            | str      |              | 取引所(TWSE、TAIFEX)                                                                                                                      |
    | 2   | Symbol                              | str      |              | 商品コード(TWSE、TAIFEX)                                                                                                                  |
    | 3   | Category                            | str      |              | 商品カテゴリ(TWSE、TAIFEX)                                                                                                                |
    | 4   | TodayRefPrice                       | str      |              | 参考価格(TAIFEX)                                                                                                                          |
    | 5   | RiseStopPrice                       | str      |              | 上昇停止価格(TWSE、TAIFEX)                                                                                                                |
    | 6   | FallStopPrice                       | str      |              | 下降停止価格(TWSE、TAIFEX)                                                                                                                |
    | 7   | ChineseName                         | str      | UTF-8        | 商品の中国語名(TWSE)                                                                                                                      |
    | 8   | PreTotalMatchQty                    | str      |              | 前回取引日の取引量(TWSE、TAIFEX)                                                                                                          |
    | 9   | PreTodayRefPrice                    | str      |              | 前回取引日の参考価格(TWSE、TAIFEX)                                                                                                        |
    | 10  | PreClosePrice                       | str      |              | 前回取引日の終値(TWSE、TAIFEX)                                                                                                            |
    | 11  | IndustryCategory                    | str      |              | "産業別コード表"に基づく産業別(TWSE)                                                                                                       |
    | 12  | StockCategory                       | str      |              | “証券別コード表”に基づく証券別(TWSE)                                                                                                       |
    | 13  | BoardRemark                         | str      |              | ボード別の注記(TWSE)                                                                                                                      |
    | 14  | ClassRemark                         | str      |              | クラス別の注記(TWSE)                                                                                                                      |
    | 15  | StockAnomalyCode                    | str      |              | "株式異常コード表"に基づく株式異常コード(TWSE)                                                                                           |
    | 16  | NonTenParValueRemark                | str      |              | 非10元の額面記号(TWSE)                                                                                                                    |
    | 17  | AbnormalRecommendationIndicator     | str      |              | 異常推奨銘柄の注記(TWSE)                                                                                                                  |
    | 18  | AbnormalSecuritiesIndicator         | str      |              | 異常推奨証券の注記(TWSE)                                                                                                                  |
    | 19  | DayTradingRemark                    | str      |              | "0"：デフォルト値 "A"：現物当日取引可能証券 "B"：現物先売り先買い証券 空白：現物取引不可証券(TWSE) |
    | 20  | TradingUnit                         | str      |              | 取引単位(TWSE)                                                                                                                            |
    | 21  | TickSize                            | str      |              | 最小変動単位(TWSE)                                                                                                                        |
    | 22  | prodKind                            | str      |              | 契約種類(TAIFEX)                                                                                                                          |
    | 23  | strikePriceDecimalLocator           | str      |              | オプション商品コードの行使価格の小数点以下桁数(TAIFEX)                                                                                    |
    | 24  | PreTotalTradingAmount               | str      |              | 前回取引日の総取引額(TWSE)                                                                                                                |
    | 25  | DecimalLocator                      | str      |              | 価格小数点桁数(TAIFEX)                                                                                                                    |
    | 26  | BeginDate                           | str      | YYYYMMDD     | 上場日(TAIFEX)                                                                                                                            |
    | 27  | EndDate                             | str      | YYYYMMDD     | 上場終了日(TAIFEX)                                                                                                                        |
    | 28  | FlowGroup                           | str      |              | 流れグループ(TAIFEX)                                                                                                                      |
    | 29  | DeliveryDate                        | str      | YYYYMMDD     | 最終決済日(TAIFEX)                                                                                                                        |
    | 30  | DynamicBanding                      | str      |              | Y:適用, N:不適用 動的価格安定(TAIFEX)                                                                                                     |
    | 31  | ContractSymbol                      | str      |              | 契約コード(TAIFEX)                                                                                                                        |
    | 32  | ContractName                        | str      |              | 契約名(TAIFEX)                                                                                                                            |
    | 33  | StockID                             | str      |              | 現物株式コード(TAIFEX)                                                                                                                    |
    | 34  | StatusCode                          | str      |              | N：正常 P：取引停止 U：上場予定 状態コード(TAIFEX)                                                                                      |
    | 35  | Currency                            | str      |              | 通貨(TAIFEX)                                                                                                                              |
    | 36  | AcceptQuoteFlag                     | str      |              | 価格の提供が可能か(TAIFEX)                                                                                                                |
    | 37  | BlockTradeFlag                      | str      |              | Y:可能 N:不可能 大口取引の可否(TAIFEX)                                                                                                    |
    | 38  | ExpiryType                          | str      |              | S:標準 W:週 単位(TAIFEX)                                                                                                                  |
    | 39  | UnderlyingType                      | str      |              | E S:個別株 現物種別(TAIFEX)                                                                                                               |
    | 40  | MarketCloseGroup                    | str      |              | "商品終値時間グループ表" 商品の終値時間(TAIFEX)                                                                                           |
    | 41  | EndSession                          | str      |              | 通常取引時間：0 取引後時間：1 取引時間(TAIFEX)                                                                                           |
    | 42  | isAfterHours                        | str      |              | 朝盤 : 0 昼盤: 1 朝昼盤識別(TAIFEX)                                                                                                       |

</details>

### ProductTick

即時取引詳細情報。

<details>
    <summary>オブジェクトの属性を展開</summary>

      | No.  | フィールド名                   | データ型  | フォーマット     | 説明                                                                                      |
      |------|--------------------------------|-----------|------------------|-------------------------------------------------------------------------------------------|
      | 1    | Exchange                       | str       |                  | 取引所(TWSE、TAIFEX)                                                                      |
      | 2    | Symbol                         | str       |                  | 商品コード(TWSE、TAIFEX)                                                                    |
      | 3    | MatchTime                      | str       | %H:%M:%S.%f       | 成行時間(取引所) (TWSE、TAIFEX)                                                           |
      | 4    | OrderBookTime                  | str       | %H:%M:%S.%f       | 五段階データ時間(取引所) (TWSE、TAIFEX)                                                   |
      | 5    | TxSeq                          | str       |                  | 取引所シーケンス番号(成交情報) (TWSE、TAIFEX)                                              |
      | 6    | ObSeq                          | str       |                  | 取引所シーケンス番号(五段階情報) (TWSE、TAIFEX)                                            |
      | 7    | IsTxTrail                      | bool      |                  | 0: 非試撮、1: 試撮 取引試撮データ(TWSE、TAIFEX)                                           |
      | 8    | Is5QTrial                      | bool      |                  | 0: 非試撮、1: 試撮 五段階試撮データ(TWSE、TAIFEX)                                           |
      | 9    | IsTrail                        | bool      |                  | 0: 非試撮、1: 試撮 試撮データ(TWSE、TAIFEX)                                                |
      | 10   | DecimalLocator                 | str       |                  | 価格小数点位置(TAIFEX)                                                                    |
      | 11   | MatchPrice                     | str       |                  | 成行価格(TWSE、TAIFEX)                                                                    |
      | 12   | MatchQty                       | str       |                  | 商品取引量(TAIFEX)                                                                          |
      | 13   | MatchPriceList                 | list      |                  | 一回の価格、多くの取引価格(TWSE、TAIFEX)                                                  |
      | 14   | MatchQtyList                   | list      |                  | 一回の価格、多くの取引量(TWSE、TAIFEX)                                                    |
      | 15   | MatchBuyCount                  | str       |                  | 累積買い注文数(TAIFEX)                                                                      |
      | 16   | MatchSellCount                 | str       |                  | 累積売り注文数(TAIFEX)                                                                      |
      | 17   | TotalMatchQty                  | str       |                  | 商品取引総量(TWSE、TAIFEX)                                                                  |
      | 18   | TotalTradingAmount             | str       |                  | 商品取引総額(TWSE、TAIFEX)                                                                  |
      | 19   | TradingUnit                    | str       |                  | 取引単位(TWSE、TAIFEX)                                                                      |
      | 20   | DayHigh                        | str       |                  | 当日の最高価格(TWSE、TAIFEX)                                                                |
      | 21   | DayLow                         | str       |                  | 当日の最低価格(TWSE、TAIFEX)                                                                |
      | 22   | RefPrice                       | str       |                  | 参考価格(TWSE)                                                                              |
      | 23   | BuyPrice                       | list      |                  | 五段階価格(買い価格) (TWSE、TAIFEX)                                                          |
      | 24   | BuyQty                         | list      |                  | 五段階価格(買い数量) (TWSE、TAIFEX)                                                          |
      | 25   | SellPrice                      | list      |                  | 五段階価格(売り価格) (TWSE、TAIFEX)                                                          |
      | 26   | SellQty                        | list      |                  | 五段階価格(売り数量) (TWSE、TAIFEX)                                                          |
      | 27   | AllMarketAmount                | str       |                  | 市場全体の取引総額(TWSE)                                                                    |
      | 28   | AllMarketVolume                | str       |                  | 市場全体の取引数量(TWSE)                                                                    |
      | 29   | AllMarketCnt                   | str       |                  | 市場全体の取引筆数(TWSE)                                                                    |
      | 30   | AllMarketBuyCnt                | str       |                  | 市場全体の買い注文数(TWSE)                                                                  |
      | 31   | AllMarketSellCnt               | str       |                  | 市場全体の売り注文数(TWSE)                                                                  |
      | 32   | AllMarketBuyQty                | str       |                  | 市場全体の買い注文量(TWSE)                                                                  |
      | 33   | AllMarketSellQty               | str       |                  | 市場全体の売り注文量(TWSE)                                                                  |
      | 34   | IsFixedPriceTransaction        | str       |                  | 固定価格取引か(TWSE)                                                                        |
      | 35   | OpenPrice                      | str       |                  | 始値(TWSE、TAIFEX)                                                                          |
      | 36   | FirstDerivedBuyPrice           | str       |                  | 最初の買い価格(TAIFEX)                                                                      |
      | 37   | FirstDerivedBuyQty             | str       |                  | 最初の買い数量(TAIFEX)                                                                      |
      | 38   | FirstDerivedSellPrice          | str       |                  | 最初の売り価格(TAIFEX)                                                                      |
      | 39   | FirstDerivedSellQty            | str       |                  | 最初の売り数量(TAIFEX)                                                                      |
      | 40   | TotalBuyOrder                  | str       |                  | 累積買い注文数(TAIFEX)                                                                      |
      | 41   | TotalBuyQty                    | str       |                  | 累積買い注文契約数(TAIFEX)                                                                  |
      | 42   | TotalSellOrder                 | str       |                  | 累積売り注文数(TAIFEX)                                                                      |
      | 43   | TotalSellQty                   | str       |                  | 累積売り注文契約数(TAIFEX)                                                                  |
      | 44   | ClosePrice                     | str       |                  | 終値(TAIFEX)                                                                                |
      | 45   | SettlePrice                    | str       |                  | 決済価格(TAIFEX)                                                                              |

</details>

### RCode

これは、価格システムが返すエラーコードを記録するためのクラスです。

<details>
  <summary>オブジェクトの属性を展開</summary>

| 値    | 名称                                 | 説明                                                                                   |
| ----- | ------------------------------------ | -------------------------------------------------------------------------------------- |
| 0     | OK                                   | 成功                                                                                   |
| 1     | SOLCLIENT_WOULD_BLOCK                | API 呼び出しがブロックされるが、リクエストは非ブロックモード                           |
| 2     | SOLCLIENT_IN_PROGRESS                | API 呼び出しが進行中（非ブロックモード）                                               |
| 3     | SOLCLIENT_NOT_READY                  | API が完了できない、オブジェクトが準備されていない（例：セッションが接続されていない） |
| 4     | SOLCLIENT_EOS                        | 次の操作がストリーム終了を返す                                                         |
| 5     | SOLCLIENT_NOT_FOUND                  | MAP でフィールドが見つからない                                                         |
| 6     | SOLCLIENT_NOEVENT                    | コンテキストに処理するイベントがない                                                   |
| 7     | SOLCLIENT_INCOMPLETE                 | API 呼び出しが部分的に完了したが、すべての要求が完了していない                         |
| 8     | SOLCLIENT_ROLLBACK                   | 取引がロールバックされたとき、Commit()がこの値を返す                                   |
| 9     | SOLCLIENT_EVENT                      | SolClient セッションイベント                                                           |
| 10    | CLIENT_ALREADY_CONNECTED             | 接続済み                                                                               |
| 11    | CLIENT_ALREADY_DISCONNECTED          | 切断済み                                                                               |
| 12    | ANNOUNCEMENT                         | アナウンスメッセージ                                                                   |
| -1    | FAIL                                 | 失敗                                                                                   |
| -2    | CONNECTION_REFUSED                   | 接続拒否                                                                               |
| -3    | CONNECTION_FAIL                      | 接続失敗                                                                               |
| -4    | ALREADY_EXISTS                       | 対象オブジェクトがすでに存在                                                           |
| -5    | NOT_FOUND                            | 対象オブジェクトが存在しない                                                           |
| -6    | CLIENT_NOT_READY                     | 接続が準備できていない                                                                 |
| -7    | USER_SUBSCRIPTION_LIMIT_EXCEEDED     | サブスクリプション数の上限を超過                                                       |
| -8    | USER_NOT_APPLIED                     | 申請していない                                                                         |
| -9    | USER_NOT_VERIFIED                    | 認証されていない                                                                       |
| -10   | USER_VERIFICATION_FAIL               | 認証失敗                                                                               |
| -11   | SUBSCRIPTION_FAIL                    | 商品サブスクリプション失敗                                                             |
| -12   | RECOVERY_FAIL                        | 回復失敗                                                                               |
| -13   | DOWNLOAD_PRODUCT_FAIL                | 基本データファイルのダウンロード失敗                                                   |
| -14   | MESSAGE_HANDLER_FAIL                 | メッセージ処理エラー                                                                   |
| -15   | FUNCTION_SUBSCRIPTION_LIMIT_EXCEEDED | 機能サブスクリプション数超過                                                           |
| -16   | USER_NOT_VERIFIED_TWSE               | TWSE で認証されていない                                                                |
| -17   | USER_NOT_VERIFIED_TAIFEX             | TAIFEX で認証されていない                                                              |
| -18   | USER_NOT_VERIFIED_TWSE_TAIFEX        | TWSE&TAIFEX で認証されていない                                                         |
| -9999 | UNKNOWN_ERROR                        | 不明なエラー                                                                           |

</details>

### MarketDataMart

これは、価格システムの開始点となるクラスで、いくつかのメソッドを定義しています。

これらのメソッドは、例えば取引情報や注文情報などのイベントをトリガーするために使用されます。

その後の使用では、対応するイベントハンドラを手動でマウントする必要があります。

| イベント                             | 説明                         |
| ------------------------------------ | ---------------------------- |
| **Fire_OnSystemEvent**               | システムメッセージ通知       |
| **Fire_MarketDataMart_ConnectState** | システム接続状態             |
| **Fire_OnUpdateBasic**               | 商品基本情報の更新           |
| **Fire_OnUpdateProductBasicList**    | 商品基本情報リストの更新     |
| **Fire_OnUpdateLastSnapshot**        | 商品最新スナップショット更新 |
| **Fire_OnMatch**                     | 商品取引情報                 |
| **Fire_OnOrderBook**                 | 商品注文情報                 |
| **Fire_OnUpdateTotalOrderQty**       | 商品注文量累計更新           |
| **Fire_OnUpdateOptionGreeks**        | オプション商品 Greeks 更新   |
| **Fire_OnUpdateOvsBasic**            | 海外商品基本情報更新         |
| **Fire_OnUpdateOvsMatch**            | 海外商品取引データ更新       |
| **Fire_OnUpdateOvsOrderBook**        | 海外商品五段階情報更新       |

### Sol_D

このクラスの入力パラメータは `MarketDataMart` クラスのインスタンスであり、直接 `MarketDataMart` インスタンスを `MasterQuoteDAPI` クラスに渡し、別のインスタンスを生成します。

`MasterQuoteDAPI` の実装を確認すると、`MarketDataMart` インスタンスを受け取った後、再度それを `SolAPI` クラスに渡し、さらに別のインスタンスを生成します。

まあ、これは彼らの実装方式ですので、とりあえず気にしません。

---

`Sol_D` クラスでは、ユーザーがよく使うメソッドがいくつか提供されています：

- `Sol_D.Login`: ログイン
- `Sol_D.DisConnect`: ログアウト
- `Sol_D.Subscribe`: 商品価格のサブスクリプション
- `Sol_D.UnSubscribe`: 商品価格のサブスクリプション解除

また、いくつかのイベントハンドラがあり、外部関数を定義した後、それらの指定された方法でマウントする必要があります：

- `Sol_D.Set_OnLogEvent`: ログインイベントハンドラの設定
- `Sol_D.Set_OnInterruptEvent`: システムイベントハンドラの設定
- `Sol_D.Set_OnLoginResultEvent_DAPI`: ログイン結果イベントハンドラの設定
- `Sol_D.Set_OnAnnouncementEvent_DAPI`: アナウンスメントイベントハンドラの設定
- `Sol_D.Set_OnVerifiedEvent_DAPI`: 検証イベントハンドラの設定
- `Sol_D.Set_OnSystemEvent_DAPI`: システムイベントハンドラの設定
- `Sol_D.Set_OnUpdateBasic_DAPI`: 商品基本情報更新イベントハンドラの設定
- `Sol_D.Set_OnMatch_DAPI`: 商品取引情報イベントハンドラの設定
