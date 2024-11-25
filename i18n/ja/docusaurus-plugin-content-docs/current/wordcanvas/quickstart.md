---
sidebar_position: 3
---

# クイックスタート

すべてのことは最初が難しいので、簡単に始められる方法を紹介します。

## 文字列から始める

まず、基本的な宣言を行い、その後すぐに使用を開始できます。

```python
from wordcanvas import WordCanvas

gen = WordCanvas()
```

デフォルト設定で、関数を呼び出すだけで文字画像を生成できます。

```python
text = "你好！Hello, World!"
img, infos = gen(text)
print(img.shape)
# >>> (67, 579, 3)
```

![sample1](./resources/sample1.jpg)

:::tip
デフォルトモードでは、出力画像のサイズは以下の要素によって決まります：

1. **フォントサイズ**：デフォルトは 64。フォントサイズが大きくなると、画像のサイズも増加します。
2. **文字の長さ**：文字が長くなるほど、画像の幅も増加します。具体的な長さは `pillow` によって決定されます。
   :::

## 特定のフォントを指定

`font` パラメータを使用して、好きなフォントを指定できます。

```python
gen = WordCanvas(
    font_path="/path/to/your/font/OcrB-Regular.ttf"
)

text = 'Hello, World!'
img, infos = gen(text)
```

![sample14](./resources/sample14.jpg)

フォントが指定された文字に対応していない場合、豆腐文字が表示されます。

```python
text = 'Hello, 中文!'
img, infos = gen(text)
```

![sample15](./resources/sample15.jpg)

:::tip
**フォントが文字をサポートしているかどうかを確認する方法：**

現在、この機能は基本的な確認方法しか提供していません。この方法は、1 回に 1 文字しかチェックできないため、すべての文字をチェックするにはループが必要です。もっと高度な確認が必要であれば、適宜拡張してください。

```python title="check_font.py"
from wordcanvas import is_character_supported, load_ttfont

target_text = 'Hello, 中文!'

font = load_ttfont("/path/to/your/font/OcrB-Regular.ttf")

for c in target_text:
    status = is_character_supported(font, c)
    if not status:
        print(f"Character: {c}, Not Supported!")

# >>> Character: 中, Not Supported!
# >>> Character: 文, Not Supported!
```

:::

## 画像サイズの設定

`output_size` パラメータを使用して、画像サイズを調整できます。

```python
gen = WordCanvas(output_size=(64, 1024)) # 高さ64、幅1024
img, infos = gen(text)
print(img.shape)
# >>> (64, 1024, 3)
```

![sample4](./resources/sample4.jpg)

指定したサイズが文字画像のサイズより小さい場合、文字画像は自動的に縮小されます。

つまり、文字が詰め込まれて、細長い長方形の形になります。例えば：

```python
text = '你好' * 10
gen = WordCanvas(output_size=(64, 512))  # 高さ64、幅512
img, infos = gen(text)
```

![sample8](./resources/sample8.jpg)

## 背景色の調整

`background_color` パラメータを使用して、背景色を調整できます。

```python
gen = WordCanvas(background_color=(255, 0, 0)) # 赤い背景
img, infos = gen(text)
```

![sample2](./resources/sample2.jpg)

## 文字色の調整

`text_color` パラメータを使用して、文字色を調整できます。

```python
gen = WordCanvas(text_color=(0, 255, 0)) # 緑色の文字
img, infos = gen(text)
```

![sample3](./resources/sample3.jpg)

## 文字の整列

:::warning
先程言及した画像サイズについて覚えておいてください。デフォルトでは、**文字の整列を設定しても意味がありません**。文字画像に余白が必要です。余白がないと整列効果を確認できません。
:::

`align_mode` パラメータを使用して、文字の整列モードを調整できます。

```python
from wordcanvas import AlignMode, WordCanvas

gen = WordCanvas(
    output_size=(64, 1024),
    align_mode=AlignMode.Center
)

text = '你好！ Hello, World!'
img, infos = gen(text)
```

- **中央揃え：`AlignMode.Center`**

  ![sample5](./resources/sample5.jpg)

- **右揃え：`AlignMode.Right`**

  ![sample6](./resources/sample6.jpg)

- **左揃え：`AlignMode.Left`**

  ![sample7](./resources/sample4.jpg)

- **分散揃え：`AlignMode.Scatter`**

  ![sample8](./resources/sample7.jpg)

  :::tip
  分散揃えモードでは、各文字が分散するのではなく、単語単位で分散されます。中国語では、単語単位は 1 文字、英語では単語単位はスペースです。

  例えば、入力テキスト「你好！Hello, World!」は次のように分割されます：

  - ["你", "好", "！", "Hello,", "World!"]

  空白を無視して、分散整列を行います。

  さらに、入力文字が 1 単語だけの場合、分散整列は中国語では中央揃えと同じ動作をし、英語の場合は単語を分割して分散整列を行います。

  使用しているロジックは以下の通りです：

  ```python
  def split_text(text: str):
      """ Split text into a list of characters. """
      pattern = r"[a-zA-Z0-9\p{P}\p{S}]+|."
      matches = regex.findall(pattern, text)
      matches = [m for m in matches if not regex.match(r'\p{Z}', m)]
      if len(matches) == 1:
          matches = list(text)
      return matches
  ```

  :::warning
  これは非常にシンプルな実装であり、すべてのニーズに対応するわけではありません。文字列の分割に関してより完全な解決策をお持ちの方は、ぜひご提供ください。
  :::

## 文字方向の調整

`direction` パラメータを使用して、文字の方向を調整できます。

- **横向き文字の出力**

  ```python
  text = '你好！'
  gen = WordCanvas(direction='ltr') # 左から右の横向き文字
  img, infos = gen(text)
  ```

  ![sample9](./resources/sample9.jpg)

- **縦向き文字の出力**

  ```python
  text = '你好！'
  gen = WordCanvas(direction='ttb') # 上から下の縦向き文字
  img, infos = gen(text)
  ```

  ![sample10](./resources/sample10.jpg)

- **縦向き文字で分散整列**

  ```python
  text = '你好！'
  gen = WordCanvas(
      direction='ttb',
      align_mode=AlignMode.Scatter,
      output_size=(64, 512)
  )
  img, infos = gen(text)
  ```

  ![sample11](./resources/sample11.jpg)

## 出力方向の調整

`output_direction` パラメータを使用して、出力方向を調整できます。

:::tip
**このパラメータの使用タイミング**：縦向き文字を出力したいが、画像を水平に表示したい場合に使用します。
:::

- **縦向き文字を水平に出力**

  ```python
  from wordcanvas import OutputDirection, WordCanvas

  gen = WordCanvas(
      direction='ttb',
      output_direction=OutputDirection.Horizontal
  )

  text = '你好！'
  img, infos = gen(text)
  ```

  ![sample12](./resources/sample12.jpg)

- **横向き文字を垂直に出力**

  ```python
  from wordcanvas import OutputDirection, WordCanvas

  gen = WordCanvas(
      direction='ltr',
      output_direction=OutputDirection.Vertical
  )

  text = '你好！'
  img, infos = gen(text)
  ```

  ![sample13](./resources/sample13.jpg)

## 文字圧縮

特定のシーンでは文字が非常に圧縮されて表示される場合があります。その場合、`text_aspect_ratio` パラメータを使用して文字の圧縮度を調整できます。

```python
gen = WordCanvas(
    text_aspect_ratio=0.25, # 文字高さ / 文字幅 = 1/4
    output_size=(32, 1024),
)  # 文字を圧縮

text="圧縮テスト"
img, infos = gen(text)
```

![sample16](./resources/sample16.jpg)

:::info
圧縮された文字のサイズが `output_size` より大きい場合、画像は自動的にスケーリングされる点に注意してください。つまり、文字が圧縮されても、最終的にスケーリングされて元のサイズに戻る場合があります。
:::

## ダッシュボード

基本的な機能はこれで紹介し終わりました。

最後に、ダッシュボードの機能を紹介します。

```python
gen = WordCanvas()
print(gen)
```

`print` を使わずに直接出力することもできます。`__repr__` メソッドを実装しているので、出力されるのはシンプルなダッシュボードです。

![dashboard](./resources/dashboard.jpg)

表示される内容：

- 最初の列は **Property** で、すべての設定パラメータ。
- 次の列は **Current Value** で、現在のパラメータ値。
- 次の列は **SetMethod** で、パラメータの設定方法。`set` と書かれているものは直接設定可能、`reinit` と書かれているものは再初期化が必要です。
- 次の列は **DType** で、パラメータのデータ型。
- 最後の列は **Description** で、パラメータの説明。

設定はほとんど直接変更できます。これにより、出力特性を変更するために新たに `WordCanvas` オブジェクトを作成する必要はなく、既存のオブジェクトを変更するだけで済みます。`reinit` が必要なパラメータは、フォント形式のような初期化が関わるものですので、それを注意してください。

```python
gen.output_size = (64, 1024)
gen.text_color = (0, 255, 0)
gen.align_mode = AlignMode.Center
gen.direction = 'ltr'
gen.output_direction = OutputDirection.Horizontal
```

設定後、再度関数を呼び出すことで新しい文字画像を得ることができます。

`reinit` 関連のパラメータを変更した場合、エラーが発生します：

- **AttributeError: can't set attribute**

  ```python
  gen.text_size = 128
  # >>> AttributeError: can't set attribute
  ```

:::danger
もちろん、強制的にパラメータを設定することもできます。Python ユーザーとして、私はあなたを止めることはできませんが：

```python
gen._text_size = 128
```

しかし、こうすると後でエラーが発生しますので、注意してください！

再初期化して新しいオブジェクトを作成するのがベストです。
:::

## まとめ

いくつかの機能はまだ紹介していませんが、基本的な機能はこれで紹介が終わりました。

次の章では、さらに進んだ機能について紹介します。
