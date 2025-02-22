---
sidebar_position: 3
---

# クイックスタート

私たちは、前処理と後処理のロジックを含むシンプルなモデル推論インターフェースを提供しています。

## モデル推論

まずは、以下のコードを実行してみてください。実行して問題なく動作するか確認してみましょう：

```python
import cv2
from skimage import io
from mrzscanner import MRZScanner

# モデルを作成
model = MRZScanner()

# オンライン画像を読み込む
img = io.imread('https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# モデル推論
result = model(img, do_center_crop=True, do_postprocess=False)

# 結果の出力
print(result)
# {
#     'mrz_polygon':
#         array(
#             [
#                 [ 158.536 , 1916.3734],
#                 [1682.7792, 1976.1683],
#                 [1677.1018, 2120.8926],
#                 [ 152.8586, 2061.0977]
#             ],
#             dtype=float32
#         ),
#     'mrz_texts': [
#         'PCAZEQAQARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#         'C946302620AZE6707297F23031072W12IMJ<<<<<<<40'
#     ],
#     'msg': <ErrorCodes.NO_ERROR: 'No error.'>
# }
```

正常に実行された場合、コードの詳細に進みましょう。

:::info
モデルの自動ダウンロード機能を設計しており、プログラムがモデルの不足を検出した際に、自動的にサーバーからダウンロードを行います。
:::

:::tip
上記の例で使用されている画像のダウンロードリンクはこちらです：[**midv2020_test_mrz.jpg**](https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg)

<div align="center" >
<figure style={{width: "30%"}}>
![test_mrz](./resources/test_mrz.jpg)
</figure>
</div>
:::

## `do_center_crop` パラメータの使用

この画像はおそらくモバイルデバイスで撮影されたもので、縦長の形状をしています。直接モデルに入力すると、文字が過度に変形する可能性があります。そこで、推論時に`do_center_crop`パラメータを追加して、画像を中央でクロップする処理を行います。

このパラメータのデフォルト値は`False`です。なぜなら、ユーザーの認識がないまま画像を変更すべきではないと考えているためです。しかし、実際のアプリケーションでは、画像は必ずしも標準の正方形サイズではありません。

実際には、画像のサイズとアスペクト比は多様です。例えば：

- モバイルで撮影された写真は一般的に 9:16 のアスペクト比；
- スキャンされた書類は A4 サイズが一般的；
- ウェブサイトのスクリーンショットは 16:9 のアスペクト比；
- ウェブカメラで撮影された画像は通常 4:3 のアスペクト比。

これらの正方形でない画像は、適切に処理せずにそのまま推論にかけると、多くの無関係な領域や空白を含むことがあり、モデルの推論結果に悪影響を与えることがあります。中央クロップを行うことで、これらの無関係な領域を減らし、画像の中心に集中することができ、推論の精度と効率を向上させることができます。

使用方法は以下の通りです：

```python
from mrzscanner import MRZScanner

model = MRZScanner()

result = model(img, do_center_crop=True) # 中央クロップを使用
```

:::tip
**使用タイミング**：『MRZ 領域を切り取らず』、かつ画像が正方形でない場合は中央クロップを使用できます。
:::

:::info
`MRZScanner`は`__call__`メソッドを使ってラップされているため、インスタンスを直接呼び出して推論を行うことができます。
:::

## `do_postprocess` パラメータの使用

中心クロップの他に、さらなる精度向上のために `do_postprocess` オプションを提供しています。

このパラメータのデフォルト値は `False` で、理由は先程と同様で、ユーザーの認識なしに認識結果を変更すべきではないと考えたからです。

実際のアプリケーションでは、MRZ 領域にはいくつかの規則が存在します。例えば：国コードは大文字のアルファベットのみ、性別は `M` または `F` のみ、日付に関するフィールドは数字だけなどです。これらの規則は MRZ 領域を規範化するために使用できます。

そこで、規範化できる領域に対して人工的に修正を加えます。以下は、数値が出現しないフィールドにおいて、誤認識された数字を正しい文字に置き換える修正の概念を実装したコード片です：

```python
import re

def replace_digits(text: str):
    text = re.sub('0', 'O', text)
    text = re.sub('1', 'I', text)
    text = re.sub('2', 'Z', text)
    text = re.sub('4', 'A', text)
    text = re.sub('5', 'S', text)
    text = re.sub('8', 'B', text)
    return text

if doc_type == 3:  # TD1
    if len(results[0]) != 30 or len(results[1]) != 30 or len(results[2]) != 30:
        return [''], ErrorCodes.POSTPROCESS_FAILED_TD1_LENGTH
    # Line1
    doc = results[0][0:2]
    country = replace_digits(results[0][2:5])
```

私たちのプロジェクトでは、この後処理が精度を大幅に向上させることはありませんでしたが、それでもこの機能を保持することで、特定の状況下では誤った認識結果を修正することができます。

`do_postprocess` を `True` に設定して推論すると、通常は結果がより良くなります：

```python
result = model(img, do_postprocess=True)
```

または、元のモデルの出力結果をそのまま確認したい場合は、デフォルト値を使用できます。

## `DocAligner` と併用

時々、`do_center_crop` を使用しても検出に失敗する場合があります。そのような場合には、`DocAligner` を使用して先に証明書の位置を見つけ、その後に MRZ 認識を行うことができます。

```python
import cv2
from docaligner import DocAligner # DocAligner をインポート
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
# {
#     'mrz_polygon':
#         array(
#         [
#             [ 34.0408 , 378.497  ],
#             [756.4258 , 385.0492 ],
#             [755.8944 , 443.63843],
#             [ 33.5094 , 437.08618]
#         ], dtype=float32
#     ),
#     'mrz_texts': [
#         'PCAZEQAQARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#         'C946302620AZE6707297F23031072W12IMJ<<<<<<<40'
#     ],
#     'msg': <ErrorCodes.NO_ERROR: 'No error.'>
# }
```

:::warning
`DocAligner` を使って前処理を行う場合、MRZ 領域はすでに一定の場所を占めている可能性があるため、その場合 `do_center_crop` を使用しない方が良いでしょう。中央クロップを行うと、MRZ 部分が切り取られてしまう可能性があります。
:::

:::tip
`DocAligner` の使い方については [**DocAligner 技術文書**](https://docsaid.org/ja/docs/docaligner/) を参照してください。
:::

## エラーメッセージ

エラーメッセージを利用者に説明するために、`ErrorCodes` クラスを設計しました。

モデル推論中にエラーが発生した場合、以下のエラーメッセージが表示されます：

```python
class ErrorCodes(Enum):
    NO_ERROR = 'No error.'
    INVALID_INPUT_FORMAT = 'Invalid input format.'
    POSTPROCESS_FAILED_LINE_COUNT = 'Postprocess failed, number of lines not 2 or 3.'
    POSTPROCESS_FAILED_TD1_LENGTH = 'Postprocess failed, length of lines not 30 when `doc_type` is TD1.'
    POSTPROCESS_FAILED_TD2_TD3_LENGTH = 'Postprocess failed, length of lines not 36 or 44 when `doc_type` is TD2 or TD3.'
```

ここでは、入力フォーマットが正しくない場合や、行数が正しくない場合などの基本的なエラーがフィルタリングされます。

## チェックディジット

チェックディジットは MRZ においてデータの正確性を保証するための重要な部分で、データ入力ミスを防ぐために使用されます。

- 詳細な手順については [**参考文献：チェックディジット**](./reference#チェックデジット) に記載されています。

---

ここで言いたいのは：

- **私たちはチェックディジットの計算機能を提供していません！**

MRZ のチェックディジット計算方法は唯一ではなく、標準的な計算方法の他にも、異なる地域の MRZ は独自の計算方法を定義しています。そのため、チェックディジットの計算方法を指定することは、ユーザーの柔軟性を制限する可能性があります。

:::info
ちょっとした豆知識をシェアします：

台湾の外国人居留証の MRZ のチェックディジットは、世界標準と異なり、政府と協力して開発されない限り、その計算方法はわかりません。
:::

私たちの目標は、MRZ 認識に特化したモデルを訓練することです。各出力はモデルが自動で形式を判断し、チェックディジットの計算機能については多くの他のオープンソースプロジェクトが提供しているので、例えば以前引用した [**Arg0s1080/mrz**](https://github.com/Arg0s1080/mrz) のプロジェクトを利用することをお勧めします。
