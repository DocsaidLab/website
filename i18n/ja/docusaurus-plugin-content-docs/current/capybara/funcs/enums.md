# Enums

OpenCV の列挙型は非常に多く、便利に使用するために、よく使われる列挙型を capybara に整理しました。これらの列挙型は、一般的なパラメータやタイプを参照するための明確で便利な方法を提供し、コードの可読性と保守性の向上に役立ちます。

ほとんどの列挙型の値は直接 OpenCV の列挙値を参照しており、これにより列挙値の一貫性が保たれます。別の列挙値を使用する必要がある場合は、OpenCV の列挙値を直接参照できます。

## 列挙型の概要

- [**INTER**](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/enums.py#L12)：異なる画像補間法を定義します。
- [**ROTATE**](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/enums.py#L20)：画像の回転角度を定義します。
- [**BORDER**](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/enums.py#L26)：境界処理方法を定義します。
- [**MORPH**](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/enums.py#L35)：形態学的操作の構造要素の形状を定義します。
- [**COLORSTR**](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/enums.py#L41)：端末で表示される色の文字列を定義します。
- [**FORMATSTR**](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/enums.py#L60)：文字列の書式設定を定義します。
- [**IMGTYP**](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/enums.py#L66)：サポートされている画像ファイルタイプを定義します。

## capybara.INTER

画像のサイズ変更や再サンプリング時に選択される補間法。

- `NEAREST`：最近傍補間。
- `BILINEAR`：二線形補間。
- `CUBIC`：三次補間。
- `AREA`：面積補間。
- `LANCZOS4`：Lanczos 補間（4 つの Lanczos 窓を使用）。

## capybara.ROTATE

画像の回転角度。

- `ROTATE_90`：時計回りに 90 度回転。
- `ROTATE_180`：180 度回転。
- `ROTATE_270`：反時計回りに 90 度回転。

## capybara.BORDER

画像の境界の拡張方法。

- `DEFAULT`：デフォルトの境界処理方法。
- `CONSTANT`：定数境界、指定した色で埋める。
- `REFLECT`：鏡面反射境界。
- `REFLECT_101`：もう一つの鏡面反射境界。
- `REPLICATE`：境界の最も外側のピクセルを複製。
- `WRAP`：ラップ境界。

## capybara.MORPH

形態学的フィルタリングで使用される構造要素の形。

- `CROSS`：十字形。
- `RECT`：矩形。
- `ELLIPSE`：楕円形。

## capybara.COLORSTR

コンソール出力の色コード。

- `BLACK`：黒。
- `RED`：赤。
- `GREEN`：緑。
- `YELLOW`：黄。
- `BLUE`：青。
- `MAGENTA`：マゼンタ。
- `CYAN`：シアン。
- `WHITE`：白。
- `BRIGHT_BLACK`：明るい黒。
- `BRIGHT_RED`：明るい赤。
- `BRIGHT_GREEN`：明るい緑。
- `BRIGHT_YELLOW`：明るい黄。
- `BRIGHT_BLUE`：明るい青。
- `BRIGHT_MAGENTA`：明るいマゼンタ。
- `BRIGHT_CYAN`：明るいシアン。
- `BRIGHT_WHITE`：明るい白。

## capybara.FORMATSTR

文字列の書式設定オプション。

- `BOLD`：太字。
- `ITALIC`：斜体。
- `UNDERLINE`：下線。

## capybara.IMGTYP

サポートされている画像ファイルタイプ。

- `JPEG`：JPEG 形式の画像。
- `PNG`：PNG 形式の画像。
