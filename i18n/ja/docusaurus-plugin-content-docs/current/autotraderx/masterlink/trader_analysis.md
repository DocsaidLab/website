---
sidebar_position: 4
---

# 注文システム分析

## MasterTradePy

元富証券の Python モジュールをダウンロードした後、以下のコマンドで注文システムをインストールします：

```powershell
pip install .\MasterLink_PythonAPI\MasterTradePy\MasterTradePy\64bit\MasterTradePy-0.0.23-py3-none-win_amd64.whl
```

:::tip
私たちが使用しているバージョンは `0.0.23` です。
:::

## 公式技術ドキュメント

- [**元富証券-注文 API**](https://mlapi.masterlink.com.tw/web_api/service/document/python-trade)
- [**公式プログラム例：sample.py**](https://github.com/DocsaidLab/AutoTraderX/blob/main/MasterLink_PythonAPI/MasterTradePy/MasterTradePy/64bit/sample.py)

## コアモジュール

元富証券の Python モジュールは以下の主要なモジュールに分解されています：

### Solace

:::tip
Solace は、高性能メッセージングミドルウェアと関連サービスを提供する会社で、その技術はリアルタイムデータストリームおよびイベント駆動型アーキテクチャに特に適しています。
:::

プログラム内で多くの `Solace` クラスを見ることができ、これらは Solace メッセージブローカーとの通信に使用されます。

1. **メッセージングサービス**

   - **MessagingService**: これはメッセージサービスの主要なインターフェースで、Solace メッセージブローカーとの接続を設定および確立するために使用されます。
   - **ReconnectionListener**: これらのインターフェースは、接続の切断と再接続のイベントを処理するために使用されます。
   - **ReconnectionAttemptListener**: これらのインターフェースは、接続の切断と再接続のイベントを処理するために使用されます。
   - **ServiceInterruptionListener**: これらのインターフェースは、接続の切断と再接続のイベントを処理するために使用されます。
   - **RetryStrategy**: このインターフェースは、メッセージ再送の戦略を定義するために使用されます。
   - **ServiceEvent**: サービスのライフサイクルに関するイベントを処理します。

2. **トピックのサブスクリプション**

   - **TopicSubscription**: 特定のトピックのメッセージをサブスクライブするために使用されます。

3. **メッセージ受信および発行者**

   - **MessageHandler**: 受信したメッセージを処理するために使用されます。
   - **InboundMessage**: 受信したメッセージを処理するために使用されます。
   - **DirectMessageReceiver**: メッセージブローカーから直接メッセージを受信するための受信機。
   - **OutboundMessage**: 送信するメッセージを表します。
   - **RequestReplyMessagePublisher**: リクエストを発行し、応答を受信するためのメッセージ発行機。
   - **PublishFailureListener**: メッセージ発行の失敗イベントを処理するために使用されます。
   - **FailedPublishEvent**: メッセージ発行の失敗イベントを処理するために使用されます。

4. **トピックおよびキャッシュ**

   - **Topic**: メッセージのトピックを表します。動的または固定のものがあります。
   - **CachedMessageSubscriptionRequest**: キャッシュされたメッセージのサブスクリプションリクエストを処理します。
   - **CacheRequestOutcomeListener**: キャッシュリクエストの結果を処理するためのインターフェースとクラスです。
   - **CacheRequestOutcome**: キャッシュリクエストの結果を処理するためのインターフェースとクラスです。

5. **ライフサイクルおよびユーティリティ**

   - **TerminationNotificationListener**: コンポーネント終了の通知とイベントを処理するために使用されます。
   - **TerminationEvent**: コンポーネント終了の通知とイベントを処理するために使用されます。

### SolClient

このクラスは、Solace メッセージングシステムを使用するための一連の機能をラップしたものです。主に接続の作成、サブスクリプションの管理、メッセージの送受信などに使用されます。

- **`__init__`**

  - **目的**: 必要な設定パラメータとユーザーの資格情報を使用して `SolClient` を初期化します。
  - **パラメータ**: `clientname`、`sol_config`、`username`（オプション）。
  - **操作**: 接続状態のデフォルト値、メッセージサービスオブジェクト、およびユーザーの資格情報を設定します。

- **`create_connection`**

  - **目的**: 提供された設定とハンドラーを使用して Solace ブローカーとの接続を確立します。
  - **パラメータ**: `message_handler` と `service_handler`。
  - **操作**：
    - セッションパラメータに基づいたブローカー属性を構築します。
    - 再接続戦略とその他の設定を備えた `MessagingService` を初期化します。
    - サービスに接続し、メッセージ受信機および発行機を設定します。
    - メッセージ受信機と発行機を起動します。
    - 接続状態をチェックし、適切な応答コードを返します。

- **`disconnect`**

  - **目的**: メッセージサービスを適切に切断し、リソースをクリーンアップします。
  - **操作**：
    - メッセージ受信機を終了します。
    - メッセージサービスの接続を切断します。
    - 接続状態フラグをリセットします。

- **`send_request`**

  - **目的**: 指定されたトピックにリクエストメッセージを送信し、応答を待機します。
  - **パラメータ**: `topic`、`msg`（メッセージ内容）。
  - **操作**：
    - メッセージを構築し、指定されたトピックに送信します。
    - 指定されたタイムアウト内で応答を待機し、処理します。
    - メッセージ送信中の例外とエラーを処理します。

- **`add_subscription`**

  - **目的**: 指定されたトピックをサブスクライブし、その上で発行されたメッセージを受信します。
  - **パラメータ**: `topic`。
  - **操作**：メッセージ受信機にトピックのサブスクリプションを追加し、エラーハンドリングを行います。

- **`remove_subscription`**

  - **目的**: トピックから既存のサブスクリプションを削除します。
  - **パラメータ**: `topic`。
  - **操作**：メッセージ受信機からトピックのサブスクリプションを削除し、エラーハンドリングを行います。

- **`request_cached_only`**

  - **目的**: キャッシュからのメッセージのみをリクエストします。
  - **パラメータ**: `cachename`、`topic`、`timeout`、`cacheId`。
  - **操作**：
    - キャッシュのみのサブスクリプションリクエストを作成します。
    - キャッシュリクエストを送信し、カスタムリスナーを使用して完了または失敗を監視します。

- **`GetRequestID`**

  - **目的**: 一意に識別される操作のために一意のリクエスト ID を生成します。
  - **操作**：タイムスタンプに基づいた ID を生成し、前回使用した ID より大きくなることを保証します。

### SorApi

このクラスは、証券注文ルーティングシステム（SORS）との通信インターフェースをカプセル化したものです。

このクラスは、ユーザーが接続、リクエスト送信、レスポンスの受信と処理を行うための一連の機能を提供します。言い換えれば、これは証券会社の注文システムに接続するための本体 API です。

この API 内のほとんどの機能は.dll ファイルにカプセル化されているため、元富証券が Windows システムのみを使用するよう制限している理由でもあります。

元々はこれらの.dll ファイルも分解しようと考えていましたが、分解してもあまり意味がないため、そのままにしておきました。

いずれにしても、他のプログラムで`OnSorConnect`のような`Sor`という言葉が含まれている場合、それはこのモジュールで機能を使用するためのものです。

### MarketTrader

これは取引の基本機能を定義した ABC クラスですが、実装はされておらず、私たちが実装する必要があります。

定義内容は以下の通りです：

```python title="MasterTradePy\64bit\MasterTradePy\model.py"
class MarketTrader(metaclass=ABCMeta):
    @abstractmethod
    def OnNewOrderReply(self, data) -> None:
        pass

    @abstractmethod
    def OnChangeReply(self, data) -> None:
        pass

    @abstractmethod
    def OnCancelReply(self, data) -> None:
        pass

    @abstractmethod
    def OnReport(self, data) -> None:
        pass

    @abstractmethod
    def OnAnnouncementEvent(self, data) -> None:
        pass

    @abstractmethod
    def OnSystemEvent(self, event: SystemEvent) -> None:
        pass

    @abstractmethod
    def OnError(self, error: MTPYError):
        pass

    @abstractmethod
    def OnReqResult(self, workid: str, data):
        pass
```

### Order

これは注文の`dataclass`クラスで、注文内のすべてのデータを定義しています。

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
私たちは株式を推奨していません。証券口座内の株式情報はすべてモザイク処理されます。
:::

これは API 呼び出し時に実際に取得される注文報告データです。

```python title="MasterTradePy\64bit\MasterTradePy\model.py"
@dataclass
class ReportOrder:
    orgOrder: Order
    order: Order
    lastMessage: str = field(init = False, default="")
    scBalance: str = field(init = False, default="")
```

`orgOrder` と `order` の 2 つの状態が含まれていますが、これは元の注文と取引所によって確認された注文を表している可能性があります。

以下はシステムから取得した委託報告と成行報告の内容を実際に示したものです：

- **1. 委託報告**

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
      tradingAccount='私はモザイク',
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
      tradingAccount='私はモザイク',
      ordNo='j0042',
      trxTime='08:31:31.926000',
      lastdealTime='',
      status='101)委託が受け入れられました（取引所が受け入れました）',
      leavesQty='2000',
      cumQty='0',
      dealPri='',
      tableName='ORD:TwsOrd'),
      lastMessage='',
      scBalance=''
    )
  )
  ```

- **2. 成行報告**

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
      tradingAccount='私はモザイク',
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
      tradingAccount='私はモザイク',
      ordNo='j0039',
      trxTime='08:31:31.926000',
      lastdealTime='09:00:11.609000',
      status='111)全ての取引が成立しました',
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
私たちのテストでは、注文データの `ReportOrder.order.dealPri` 欄が取引後も空白のままであるため、取引価格を確認することができません。

これはバグの可能性がありますが、確定はできません。
:::

### MasterTradeAPI

これは最外層の API で、ユーザーが直接注文操作を行うために使用します。

ここでは、元富証券が提供するサンプルコードに従って、[**sample.py**](https://github.com/DocsaidLab/AutoTraderX/blob/main/MasterLink_PythonAPI/MasterTradePy/MasterTradePy/64bit/sample.py)で使用方法を簡単に確認します：

- **Step 1: `MarketTrader` クラスのインスタンスを作成し、入力パラメータとして使用**

  ```python
  api = MasterTradeAPI(MarketTrader())
  ```

- **Step 2: 元富証券の注文システムに接続**

  ```python
  # 接続先は固定
  api.SetConnectionHost('solace140.masterlink.com.tw:55555')

  # アカウントとパスワードを入力し、ログイン状態を指定
  # is_sim: シミュレーションサーバーに接続するかどうか。Trueはシミュレーションサーバーに接続、Falseは本番サーバーに接続
  # is_force: 強制ログインするかどうか。Trueは強制ログイン、Falseは通常ログイン
  # is_event: 特定のイベント（コンテストなど）にログインするかどうか。Trueはイベントを有効、Falseは無効
  rc = api.Login(username, password, is_sim, is_force, is_event)
  ```

  ログインプロセスでは、`SolClient` を使用してシステムメッセージの接続を確立し、`SorApi` を使用して注文システムの接続を確立します。

  ログイン後には二要素認証が行われます：

  ```python
  accounts = [x[4:] for x in api.accounts]
  rcc = api.CheckAccs(tradingAccounts=accounts)
  ```

  この二要素認証はリスト方式を採用しており、この API が複数のアカウントに対応していることがわかります。

  つまり、`api.Login` を繰り返し呼び出して異なるアカウントにログインし、取引時にどのアカウントを使用して注文を行うかを指定することができます。

- **Step 3: 注文**

  注文時には、`Order` オブジェクトを設定し、`api.NewOrder` を呼び出して注文を行います。

  ```python
  from MasterTradePy.model import *

  symbol = input(u'購入する株式の銘柄コードを入力してください:')
  api.ReqBasic(symbol)
  account = input(u'注文アカウントを入力してください:')
  price = input(u'購入する株式の価格を入力してください（空白で市場価格注文）:')
  qty = input(u'購入する株式の数量を入力してください（1単位は1000株）:')
  orderTypeSymbol = input(u'注文の種類を入力してください（I:IOC, F:FOK, その他:ROD）:')

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
      print(u'注文が送信されました')
  else:
      print(u'注文失敗！プログラムを再実行し、報告内容に基づいて入力を修正してください')
  ```

- **Step 4: 価格変更**

  `OrderPriceChange` を呼び出して注文番号を設定し、`api.ChangeOrderPrice` を使用して価格を変更します。

  ```python
  from MasterTradePy.model import *

  ordNo = input(u'注文番号を入力してください:')
  account = input(u'注文アカウントを入力してください:')
  price = input(u'株式の価格を入力してください（空白で市場価格注文）:')
  replaceOrder = OrderPriceChange(ordNo=ordNo, price=price,tradingAccount=account)

  rcode = api.ChangeOrderPrice(replaceOrder)
  if rcode == RCode.OK:
      print(u'注文が送信されました')
  else:
      print(u'価格変更失敗！プログラムを再実行し、報告内容に基づいて入力を修正してください')
  ```

- **Step 5: 数量変更**

  `OrderQtyChange` を呼び出して注文番号を設定し、`api.ChangeOrderQty` を使用して数量を変更します。

  ```python
  from MasterTradePy.model import *

  ordNo = input(u'注文番号を入力してください:')
  account = input(u'注文アカウントを入力してください:')
  qty = input(u'株式の数量を入力してください（1単位は1000株）:')
  replaceOrder = OrderQtyChange(ordNo=ordNo, qty=qty, tradingAccount=account)

  rcode = api.ChangeOrderQty(replaceOrder)
  if rcode == RCode.OK:
      print(u'注文が送信されました')
  else:
      print(u'数量変更失敗！プログラムを再実行し、報告内容に基づいて入力を修正してください')
  ```

- **Step 6: その他の機能**

  - **`api.QryRepAll`**: すべての注文を照会
  - **`api.QryRepDeal`**: 成約報告を照会
  - **`api.ReqInventoryOpen`**: 初期在庫を照会
  - **`api.ReqInventoryRayinTotal`**: 在庫を照会
  - **`api.QrySecInvQty_Rayin`**: 証券源を照会
  - **`api.QryProdCrQty_Rayin`**: 資券配分を照会
