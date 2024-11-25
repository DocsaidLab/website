---
sidebar_position: 3
---

# Enums

OpenCV には多くの列挙型（Enum）が存在します。これらの中からよく使われるものを整理し、`docsaidkit` に取り込んでいます。これらの列挙型は、よく使われるパラメータやタイプを明確で便利に参照できる方法を提供し、コードの可読性や保守性を向上させます。

ほとんどの列挙型の値は OpenCV の列挙値をそのまま使用しており、一貫性を保っています。もし OpenCV の他の列挙値を使用したい場合は、直接 OpenCV の列挙値を参照することができます。

## 列挙型クラスの概要

- [**INTER**](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/enums.py#L12)：画像の補間方法を定義。
- [**ROTATE**](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/enums.py#L20)：画像の回転角度を定義。
- [**BORDER**](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/enums.py#L26)：画像の境界処理方法を定義。
- [**MORPH**](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/enums.py#L35)：形態学的操作に使用するカーネルの形状を定義。
- [**COLORSTR**](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/enums.py#L41)：コンソール出力の色を定義。
- [**FORMATSTR**](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/enums.py#L60)：文字列のフォーマットオプションを定義。
- [**IMGTYP**](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/enums.py#L66)：サポートされている画像ファイルタイプを定義。

## docsaidkit.INTER

画像リサイズやリサンプリング時に使用する補間方法。

- `NEAREST`：最近傍補間。
- `BILINEAR`：バイリニア補間。
- `CUBIC`：三次補間。
- `AREA`：面積補間。
- `LANCZOS4`：Lanczos 補間（4×Lanczos 窓を使用）。

## docsaidkit.ROTATE

画像の回転角度。

- `ROTATE_90`：時計回りに 90 度回転。
- `ROTATE_180`：180 度回転。
- `ROTATE_270`：反時計回りに 90 度回転。

## docsaidkit.BORDER

画像の境界拡張方法。

- `DEFAULT`：デフォルトの境界処理。
- `CONSTANT`：定数境界（指定した色で埋める）。
- `REFLECT`：反射境界。
- `REFLECT_101`：別の種類の反射境界。
- `REPLICATE`：境界の最外周ピクセルを複製する。
- `WRAP`：包み込み境界。

## docsaidkit.MORPH

形態学的フィルタリング時に使用する構造要素の形状。

- `CROSS`：十字形。
- `RECT`：矩形。
- `ELLIPSE`：楕円形。

## docsaidkit.COLORSTR

コンソール出力用のカラーコード。

- `BLACK`：黒色。
- `RED`：赤色。
- `GREEN`：緑色。
- `YELLOW`：黄色。
- `BLUE`：青色。
- `MAGENTA`：マゼンタ色。
- `CYAN`：シアン色。
- `WHITE`：白色。
- `BRIGHT_BLACK`：明るい黒色。
- `BRIGHT_RED`：明るい赤色。
- `BRIGHT_GREEN`：明るい緑色。
- `BRIGHT_YELLOW`：明るい黄色。
- `BRIGHT_BLUE`：明るい青色。
- `BRIGHT_MAGENTA`：明るいマゼンタ色。
- `BRIGHT_CYAN`：明るいシアン色。
- `BRIGHT_WHITE`：明るい白色。

## docsaidkit.FORMATSTR

文字列のフォーマットオプション。

- `BOLD`：太字。
- `ITALIC`：斜体。
- `UNDERLINE`：下線。

## docsaidkit.IMGTYP

サポートされている画像ファイルタイプ。

- `JPEG`：JPEG 形式の画像。
- `PNG`：PNG 形式の画像。
