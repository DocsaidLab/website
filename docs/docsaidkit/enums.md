---
sidebar_position: 5
---

# enums

在 OpenCV 裡的列舉類別實在太多了，為了方便使用，我們將一些常用的列舉類別整理到 docsaidkit 中。這些列舉類別提供了一種清晰和方便的方式來引用常用的參數和類型，有助於提高程式碼的可讀性和可維護性。

大多數的列舉的數值都直接引用 OpenCV 的列舉值，這樣可以保證列舉值的一致性。如果你需要使用其他列舉值，可以直接引用 OpenCV 的列舉值。

## 列舉類別概述

- [**INTER**](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/enums.py#L12)：定義了不同的影像內插法。
- [**ROTATE**](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/enums.py#L20)：定義了影像的旋轉角度。
- [**BORDER**](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/enums.py#L26)：定義了邊界處理的方式。
- [**MORPH**](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/enums.py#L35)：定義了形態學操作的核形狀。
- [**COLORSTR**](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/enums.py#L41)：定義了終端顯示的顏色字串。
- [**FORMATSTR**](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/enums.py#L60)：定義了文字格式化的字串。
- [**IMGTYP**](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/enums.py#L66)：定義了支援的映像檔類型。

## INTER

用於影像大小調整或重採樣時選擇的內插法。

- `NEAREST`：最近鄰插值。
- `BILINEAR`：雙線性內插。
- `CUBIC`：三次內插。
- `AREA`：面積內插。
- `LANCZOS4`：Lanczos內插（使用4個Lanczos視窗）。

## ROTATE

圖像旋轉的具體角度。

- `ROTATE_90`：順時針旋轉90度。
- `ROTATE_180`：旋轉180度。
- `ROTATE_270`：逆時針旋轉90度。

## BORDER

影像邊界的擴展方式。

- `DEFAULT`：預設邊界處理方式。
- `CONSTANT`：常數邊界，使用特定顏色填滿。
- `REFLECT`：鏡像反射邊界。
- `REFLECT_101`：另一種類型的鏡像反射邊界。
- `REPLICATE`：複製最邊緣像素的邊界。
- `WRAP`：包裝邊界。

## MORPH

形態學濾波時所使用的結構元素的形狀。

- `CROSS`：交叉形。
- `RECT`：矩形。
- `ELLIPSE`：橢圓形。

## COLORSTR

用於控制台輸出的顏色代碼。

- `BLACK`：黑色。
- `RED`：紅色。
- `GREEN`：綠色。
- `YELLOW`：黃色。
- `BLUE`：藍色。
- `MAGENTA`：品紅色。
- `CYAN`：青色。
- `WHITE`：白色。
- `BRIGHT_BLACK`：亮黑色。
- `BRIGHT_RED`：亮紅色。
- `BRIGHT_GREEN`：亮綠色。
- `BRIGHT_YELLOW`：亮黃色。
- `BRIGHT_BLUE`：亮藍色。
- `BRIGHT_MAGENTA`：亮品紅色。
- `BRIGHT_CYAN`：亮青色。
- `BRIGHT_WHITE`：亮白色。

## FORMATSTR

文字格式化選項。

- `BOLD`：加粗。
- `ITALIC`：斜體。
- `UNDERLINE`：底線。

## IMGTYP

支援的圖像檔案類型。

- `JPEG`：JPEG 格式影像。
- `PNG`：PNG 格式圖片。
