---
sidebar_position: 3
---

# クイックスタート

私たちは、前後処理のロジックを含むシンプルなモデル推論インターフェースを提供しています。

まず、必要な依存関係をインポートし、`MRZScanner`クラスを作成する必要があります。

## モデル推論

以下は、`MRZScanner`を使ってモデル推論を行う簡単な例です：

```python
from mrzscanner import MRZScanner

model = MRZScanner()
```

モデルを起動したら、次に推論に使用する画像を準備します：

:::tip
`MRZScanner`が提供するテスト画像を使用できます：

ダウンロードリンク：[**midv2020_test_mrz.jpg**](https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg)

<div align="center" >
<figure style={{width: "50%"}}>
![test_mrz](./resources/test_mrz.jpg)
</figure>
</div>
:::

```python
import docsaidkit as D

img = D.imread('path/to/run_test_card.jpg')
```

または、URL から直接読み込むこともできます：

```python
import cv2
from skimage import io

img = io.imread('https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
```

この画像は長すぎるため、直接推論すると過度な文字の歪みが発生する可能性があります。そのため、モデルを呼び出す際に`do_center_crop`パラメータを有効にします。

次に、`model`を使って推論を実行します：

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
`MRZScanner`は`__call__`でラップされているため、インスタンスを直接呼び出して推論できます。
:::

:::info
初めて`MRZScanner`を使用する際には、自動でモデルをダウンロードする機能も実装しています。
:::

## `DocAligner`との併用

上記の出力結果を見て、`do_center_crop`を使用しても、いくつかの誤字があることに気づきます。

これは、全画像をスキャンしたため、画像内の文字に誤認識が生じる可能性があるためです。

精度を向上させるために、`DocAligner`を追加して MRZ 区画を整列させます：

```python
import cv2
from docaligner import DocAligner # DocAlignerをインポート
from mrzscanner import MRZScanner
from skimage import io

model = MRZScanner()

doc_aligner = DocAligner()

img = io.imread(
    'https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

flat_img = doc_aligner(img).doc_flat_img # MRZ区画を整列
print(model(flat_img))
# >>> ('PCAZEQAQARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#      'C946302620AZE6707297F23031072W12IMJ<<<<<<<40')
```

`DocAligner`を使用した後は、もう`do_center_crop`パラメータを使う必要はありません。

これで、より正確な結果が得られ、この画像の MRZ 区画が正しく認識されたことがわかります。

## エラーメッセージ

ユーザーがエラーの原因を理解できるように、`ErrorCodes`クラスを設計しました。

モデル推論でエラーが発生した場合、以下の範囲でエラーメッセージが表示されます：

```python
class ErrorCodes(Enum):
    NO_ERROR = 'No error.'
    INVALID_INPUT_FORMAT = 'Invalid input format.'
    POSTPROCESS_FAILED_LINE_COUNT = 'Postprocess failed, number of lines not 2 or 3.'
    POSTPROCESS_FAILED_TD1_LENGTH = 'Postprocess failed, length of lines not 30 when `doc_type` is TD1.'
    POSTPROCESS_FAILED_TD2_TD3_LENGTH = 'Postprocess failed, length of lines not 36 or 44 when `doc_type` is TD2 or TD3.'
```

ここでは、基本的なエラー（例：不正な入力形式や行数が正しくないなど）をフィルタリングします。

## チェックデジット

チェックデジットは、MRZ 内でデータの正確性を確保するための重要な部分です。これは、数字の正確性を検証し、データ入力エラーを防ぐために使用されます。

- 詳細な操作手順は、[**参考文献：チェックデジット**](./reference#チェックデジット)に記載しています。

---

このセクションで言いたいのは：

- **チェックデジット計算機能は提供していません！**

MRZ のチェックデジット計算方法は唯一ではなく、正規の計算方法に加えて、異なる地域の MRZ では独自のチェックデジット計算方法を再定義することができます。そのため、チェックデジット計算方法を指定すると、ユーザーの柔軟性を制限する可能性があります。

:::info
冷知識を一つ：

台湾の外国人居留証の MRZ のチェックデジットは、世界の標準とは異なります。政府と協力して開発しない限り、このチェックデジットの計算方法は分かりません。
:::

私たちの目標は、MRZ 認識に特化したモデルを訓練することです。モデルは自動的に形式を判定し、チェックデジットの計算機能は他の多くのオープンソースプロジェクトで提供されています。例えば、[**Arg0s1080/mrz**](https://github.com/Arg0s1080/mrz)ではチェックデジット計算方法が提供されているので、ユーザーはこのプロジェクトを直接利用することをお勧めします。
