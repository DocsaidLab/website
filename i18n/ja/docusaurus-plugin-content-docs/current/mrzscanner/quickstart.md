---
sidebar_position: 3
---

# クイックスタート

私たちは、前処理と後処理のロジックを含むシンプルなモデル推論インターフェースを提供しています。

まず、必要な依存関係をインポートし、`MRZScanner` クラスを作成する必要があります。

## モデル推論

:::info
自動的にモデルをダウンロードする機能を設計しています。プログラムがモデルが不足していることを検出すると、サーバーに接続して自動的にダウンロードが行われます。
:::

以下は簡単な例です：

```python
import cv2
from skimage import io
from mrzscanner import MRZScanner

# モデルの構築
model = MRZScanner()

# 画像を読み込む
img = io.imread('https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# 推論
result_mrz, error_msg = model(img)

# MRZブロックの2行の文字とエラーメッセージを出力
print(result_mrz)
# >>> ('PCAZEQAQARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#     'C946302620AZE6707297F23031072W12IMJ<<<<<<<40')
print(error_msg)
# >>> <ErrorCodes.NO_ERROR: 'No error.'>
```

:::tip
上記の例で使用された画像のダウンロードリンクはこちらです：[**midv2020_test_mrz.jpg**](https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg)

<div align="center" >
<figure style={{width: "30%"}}>
![test_mrz](./resources/test_mrz.jpg)
</figure>
</div>
:::

## `do_center_crop` パラメーターを使用する

この画像はモバイルデバイスで撮影されたもので、形が細長いため、直接モデルに渡すと文字が歪んでしまいます。そのため、モデルを呼び出す際に `do_center_crop` パラメーターを同時に有効にする方法は次の通りです：

```python
from mrzscanner import MRZScanner

model = MRZScanner()

result, msg = model(img, do_center_crop=True)
print(result)
# >>> ('PCAZEQAOARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#      'C946302620AZE6707297F23031072W12IMJ<<<<<<<40')
print(msg)
# >>> <ErrorCodes.NO_ERROR: 'No error.'>
```

:::tip
`MRZScanner` は `__call__` でラップされているため、インスタンスを直接呼び出して推論を行うことができます。
:::

## `DocAligner` と併用する

上記の出力結果をよく見ると、`do_center_crop` を行ったにもかかわらず、いくつかの誤字が見受けられます。

これは、全体画像スキャンを使用したため、モデルが画像内の文字を誤認識する可能性があるためです。

精度を向上させるために、`DocAligner` を追加して MRZ ブロックを整列させる手順は次の通りです：

```python
import cv2
from docaligner import DocAligner # DocAlignerをインポート
from mrzscanner import MRZScanner
from capybara import imwarp_quadrangle
from skimage import io

model = MRZScanner()

doc_aligner = DocAligner()

img = io.imread('https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

polygon = doc_aligner(img)
flat_img = imwarp_quadrangle(img, polygon, dst_size=(800, 480))

print(model(flat_img))
# >>> ('PCAZEQAQARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#      'C946302620AZE6707297F23031072W12IMJ<<<<<<<40')
```

`DocAligner` を使用した後は、`do_center_crop` パラメーターを使用する必要はありません。

これで、出力結果がより正確になり、画像の MRZ ブロックが正しく識別されました。

## エラーメッセージ

ユーザーがエラーの原因を理解できるように、`ErrorCodes` クラスを設計しました。

モデルの推論でエラーが発生した場合、エラーメッセージが表示され、その内容は次のようになります：

```python
class ErrorCodes(Enum):
    NO_ERROR = 'No error.'
    INVALID_INPUT_FORMAT = 'Invalid input format.'
    POSTPROCESS_FAILED_LINE_COUNT = 'Postprocess failed, number of lines not 2 or 3.'
    POSTPROCESS_FAILED_TD1_LENGTH = 'Postprocess failed, length of lines not 30 when `doc_type` is TD1.'
    POSTPROCESS_FAILED_TD2_TD3_LENGTH = 'Postprocess failed, length of lines not 36 or 44 when `doc_type` is TD2 or TD3.'
```

ここでは、入力フォーマットが不正、行数が不正など、基本的なエラーをフィルタリングします。

## チェックディジット

チェックディジット（Check Digit）は、MRZ においてデータの正確性を確認するための重要な部分であり、数字の正確性を検証して、データ入力ミスを防止します。

- 詳しい操作手順については、[**参考文献：チェックディジット**](./reference#チェックディジット)をご覧ください。

---

このセクションでは、次のことを説明します：

- **チェックディジット計算機能は提供していません！**

MRZ のチェックディジット計算方法は一意ではなく、正規の計算方法に加えて、異なる地域の MRZ では独自に計算方法を定義することがあります。そのため、特定のチェックディジット計算方法を提供すると、ユーザーの柔軟性が制限される可能性があります。

:::info
ちなみに、冷知識を一つ：

台湾の外国人居留証の MRZ チェックディジットは世界標準と異なり、政府と協力して開発しない限り、その計算方法は分かりません。
:::

私たちの目標は、MRZ 認識に特化したモデルを訓練することです。モデルの出力は自動的に形式を判定します。チェックディジット計算機能は多くの他のオープンソースプロジェクトで提供されており、例えば以前引用した [**Arg0s1080/mrz**](https://github.com/Arg0s1080/mrz) などでチェックディジット計算方法が提供されています。ユーザーはこのプロジェクトを直接使用することをお勧めします。
