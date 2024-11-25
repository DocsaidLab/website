---
sidebar_position: 6
---

# MRZ 生成

:::tip
この機能はバージョン 0.5.0 で追加されました。
:::

`WordCanvas` の開発が完了した後、このツールを使って他のこともできます。

この章では、「**機械可読ゾーン（MRZ, Machine Readable Zone）**」を生成する機能を開発しました。

## MRZ とは？

MRZ（Machine Readable Zone、機械可読ゾーン）は、パスポート、ビザ、身分証明書などの旅行書類に含まれる特定の区域で、この区域内の情報は機械によって迅速に読み取ることができます。MRZ は国際民間航空機関（ICAO）第 9303 号の規定に基づいて設計および生成され、国境での検査を迅速化し、情報処理の精度を向上させるために使用されます。

- [**ICAO Doc9309**](https://www.icao.int/publications/Documents/9303_p1_cons_en.pdf)

MRZ の構造は書類の種類によって異なり、主に以下の種類があります：

1. **TD1（身分証明書など）**

   - 3 行、各行 30 文字で構成され、合計 90 文字。
   - 含まれる情報：証明書タイプ、国コード、証明書番号、出生日、性別、有効期限、国籍、姓、名前、任意のデータ 1、任意のデータ 2。

2. **TD2（パスポートカードなど）**

   - 2 行、各行 36 文字で構成され、合計 72 文字。
   - 含まれる情報：証明書タイプ、国コード、姓、名前、証明書番号、国籍、出生日、性別、有効期限、任意のデータ。

3. **TD3（パスポートなど）**

   - 2 行、各行 44 文字で構成され、合計 88 文字。
   - 含まれる情報：証明書タイプ、国コード、姓、名前、証明書番号、国籍、出生日、性別、有効期限、任意のデータ。

4. **MRVA（ビザタイプ A）**

   - 2 行、各行 44 文字で構成され、合計 88 文字。
   - 含まれる情報：証明書タイプ、国コード、姓、名前、証明書番号、国籍、出生日、性別、有効期限、任意のデータ。

5. **MRVB（ビザタイプ B）**
   - 2 行、各行 36 文字で構成され、合計 72 文字。
   - 含まれる情報：証明書タイプ、国コード、姓、名前、証明書番号、国籍、出生日、性別、有効期限、任意のデータ。

## 合成画像

MRZ 検出モデルをトレーニングするには、大量のデータセットが必要ですが、これらのデータには個人情報が含まれているため、収集が困難です。この問題を解決するために、`WordCanvas` を使って MRZ 画像を合成することができます。

関連機能はすでに実装されているので、以下の例に従って `MRZGenerator` を呼び出してください：

```python
import docsaidkit as D
from wordcanvas import MRZGenerator

mrz_gen = MRZGenerator(
    text_color=(0, 0, 0),
    background_color=(255, 255, 255),
    interval=None,
    delimiter='&',
)
```

この設定では、文字色、背景色、区切り文字を手動で指定できます。MRZ の文字は 2〜3 行なので、区切り文字は各行の文字を区別するために使用され、デフォルトは `&` です。

設定が完了したら、関数として呼び出すだけです。`__call__` メソッドが実装されています：

```python
output_infos = mrz_gen()
```

これで、合成された MRZ 画像が得られます。出力フォーマットは以下の通りです：

- `typ`：MRZ タイプ。
- `text`：MRZ 文字。
- `points`：各文字の座標。
- `image`：MRZ 画像。

パラメータを指定しない場合、MRZ タイプ（TD1、TD2、TD3）はランダムに決定され、その後、MRZ 文字がランダムに生成され、画像が合成されます。

以下のように画像を出力できます：

```python
D.imwrite(output_infos['image'])
```

![mrz_output](./resources/mrz_output.jpg)

## 各文字の座標の表示

各文字の位置に興味があるかもしれません。これにより、文字検出モデルのトレーニングに役立ちます。この機能も提供されており、`points` プロパティを呼び出すことで取得できます。以下のコードを使って、座標を表示することができます：

```python
import cv2
import docsaidkit as D
from wordcanvas import MRZGenerator


def draw_points(img, points, color=(0, 255, 0), radius=5):
    for point in points:
        cv2.circle(img, point, radius, color, -1)
    return img


mrz_gen = MRZGenerator(
    text_color=(0, 0, 0),
    background_color=(255, 255, 255),
    interval=None,
    delimiter='&',
)

output_infos = mrz_gen()

img = draw_points(results['image'], results['points'])
D.imwrite(img)
```

![mrz_points](./resources/mrz_points.jpg)

## 文字の背景色と間隔の調整

`text_color` と `background_color` パラメータを使って、文字と背景の色を変更できます：

```python
mrz_gen = MRZGenerator(
    text_color=(255, 0, 0),
    background_color=(0, 127, 127),
)

output_infos = mrz_gen()
D.imwrite(output_infos['image'])
```

![mrz_color](./resources/mrz_color.jpg)

---

`interval` パラメータを使って、文字の間隔を調整できます：

```python
mrz_gen = MRZGenerator(
    interval=100,
)

output_infos = mrz_gen()
D.imwrite(output_infos['image'])
```

![mrz_interval](./resources/mrz_interval.jpg)

## 指定された MRZ 文字の生成

指定した MRZ 文字を使いたい場合は、`mrz_type` と `mrz_text` パラメータを渡すことができます。関数内では、文字の長さとタイプが一致するかどうかを簡単にチェックします。

:::warning
ハッシュチェックは行っていませんので、この機能は実際の有効な MRZ 文字を必要としません。
:::

```python
mrz_gen = MRZGenerator(interval=32)

output_infos = mrz_gen(
    mrz_type='TD1',
    mrz_text=[
        "I<SWE59000002<8198703142391<<<",
        "8703145M1701027SWE<<<<<<<<<<<8",
        "SPECIMEN<<SVEN<<<<<<<<<<<<<<<<"
    ]
)

D.imwrite(output_infos['image'])
```

![mrz_assign_text](./resources/mrz_assign_text.jpg)

## 参考文献

- [**Arg0s1080/mrz**](https://github.com/Arg0s1080/mrz)
- [**Detecting machine-readable zones in passport images**](https://pyimagesearch.com/2015/11/30/detecting-machine-readable-zones-in-passport-images/)
- [**ultimateMRZ-SDK**](https://github.com/DoubangoTelecom/ultimateMRZ-SDK)
- [**QKMRZScanner**](https://github.com/Mattijah/QKMRZScanner)
- [**PassportScanner**](https://github.com/evermeer/PassportScanner)
