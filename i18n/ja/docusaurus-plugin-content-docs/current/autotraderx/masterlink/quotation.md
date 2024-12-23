---
sidebar_position: 6
---

# 価格情報システム

元富証券の価格情報システム Python API を分析した後、私たちは自分のニーズに基づいて価格情報システムを開発することができます。

元富証券が提供する価格情報システムの使用例では、複数の株式を購読する方法が特に示されていませんが、私たちはソースコード内でこの機能を発見しました。モバイルアプリの実装ではなく、UI 画面に接続できないため、表示方法はコマンドラインだけになります。そのため、設計を変更して、各株式の購読情報を外部ファイルに出力することにしました。これにより、コマンドラインで情報が混乱するのを避けることができます。

## アカウントへのログイン

アカウントのユーザー名とパスワードをクラスの入力に直接記述することもできますが、私たちの方法を参考にして、アカウント情報を保存するために yaml ファイルを使用することもできます。

パラメータファイルには、アカウントのユーザー名、パスワード、アカウント番号が必要です。この情報で元富証券のアカウントにログインできます。

次に、`autotraderx` から `QuotationSystem` クラスをインポートします：

```python
from autotraderx import load_yaml
from autotraderx.masterlink import QuotationSystem

# アカウント情報の読み込み
cfg = load_yaml(DIR / "account.yaml")

# アカウントにログイン
handler = QuotationSystem(
    user=cfg["user"],
    password=cfg["password"],
    subscribe_list=["2330", "2454"]
)
```

ログイン時には、購読したい株式コードも一緒に渡します。この部分も yaml ファイルに記述しておけば、毎回プログラムを変更する必要がありません。

## 価格情報システムの起動

アカウントにログイン後、次のように価格情報システムを起動できます：

```python
handler.run()
```

![run_quotation](./img/run_quotation.jpg)

起動後、価格情報システムは実行ディレクトリにファイルを作成します。これには次の内容が含まれます：

1. **log\_[当日の日付]\_[株式コード]\_info.md**：株式の前日の終値、最新の取引価格、最新の取引量などの情報を記録します。
2. **log\_[当日の日付]\_[株式コード]\_match.md**：株式の各ティックの取引情報を記録します。

その後、何もする必要はなく、価格情報システムが自動的に株式情報を更新します。あなたは待機するだけです。

プログラムを終了するには、`Ctrl + C` を押して価格情報システムを停止できます。

## 内容例

以下に更新されたファイル内容の例を示します：

### 商品基本情報

| 項目               | 数値      |
| ------------------ | --------- |
| 中国名             | 台積電    |
| 取引所コード       | TWS       |
| 参考価格           | 921.0000  |
| 上昇制限価格       | 1010.0000 |
| 下落制限価格       | 829.0000  |
| 前営業日の取引量   | 26262     |
| 前営業日の参考価格 | 922.0000  |
| 前営業日の終値     | 921.0000  |
| 業種               | 24        |
| 株式異常コード     | 0         |
| 非 10 元面額メモ   |           |
| 異常推奨個別株メモ |           |
| 現物株当日取引メモ | A         |
| 取引単位           | 1000      |

### 日次取引データ

| 取引時間        | 取引価格 | 上昇/下落 | 取引量 | 累計  |
| --------------- | -------- | --------- | ------ | ----- |
| 11:14:28.097382 | 944.0000 | +23.000   | 2      | 23491 |
| 11:14:33.153135 | 944.0000 | +23.000   | 1      | 23492 |
| 11:14:37.089803 | 944.0000 | +23.000   | 2      | 23494 |
| 11:14:38.663758 | 944.0000 | +23.000   | 4      | 23498 |
| 11:14:59.809925 | 945.0000 | +24.000   | 1      | 23499 |
| 11:15:00.081727 | 944.0000 | +23.000   | 2      | 23501 |
| 11:15:00.196828 | 944.0000 | +23.000   | 1      | 23502 |
| 11:15:00.567548 | 944.0000 | +23.000   | 1      | 23503 |
| 11:15:04.071329 | 944.0000 | +23.000   | 1      | 23504 |
| 11:15:04.598060 | 944.0000 | +23.000   | 1      | 23505 |
| 11:15:07.634295 | 944.0000 | +23.000   | 3      | 23508 |
| 11:15:10.137589 | 944.0000 | +23.000   | 2      | 23510 |
| 11:15:12.460697 | 944.0000 | +23.000   | 3      | 23513 |

## その他の機能

現在、いくつかの機能が完成待ちです：

1. **出力形式の指定**：現在、出力先は Markdown ファイルのみですが、将来的には CSV や JSON などの他の形式に出力できるようにすることが検討されています。
2. **取引イベントのインポート**：価格情報システムと取引システムを統合し、価格情報システムが特定の価格を検出した場合に自動的に注文を出すことができるようにする機能。

出力形式については、今のところ他に特別な要求がないので、現時点では保留にしています。取引イベントのインポートについては、通常「取引戦略」と呼ばれるもので、対象によって異なるため、どのように設計するかが次の開発の重点となります。

もちろん、その他の機能も追加できます。ご提案があればお聞かせください。
