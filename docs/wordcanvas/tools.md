---
sidebar_position: 6
---

# 相關資源

文本合成工具主要用於自動生成圖像數據集，尤其是在需要大量標注數據來訓練深度學習模型的情況下。這些工具通過在圖像中嵌入合成文字來模擬真實世界中文字的出現，從而增強模型對於不同環境、字體、顏色和背景的適應性。

以下列出了目前一些較為知名的文本合成工具：

## SynthText

- [**SynthText**](https://github.com/ankush-me/SynthText)

SynthText 是一個開源專案，由 Ankush Gupta、Andrea Vedaldi 和 Andrew Zisserman 於 2016 年的 CVPR（計算機視覺和模式辨識會議）上發表。該專案旨在生成合成文本圖像，以便用於自然圖像中的文本定位研究。這些合成圖像主要用於訓練機器學習模型，特別是在光學字元辨識和文本檢測領域。

SynthText 利用自然場景圖像作為背景，並在其中合成看似自然的文本。這一過程需要背景圖像的深度和分割信息。專案的主要依賴包括 pygame、opencv、PIL、numpy、matplotlib、h5py 和 scipy。

## SynthText3D

- [**SynthText3D**](https://github.com/MhLiao/SynthText3D)

SynthText3D 基於 Unreal Engine 4.16 和 UnrealCV 插件，通過在虛擬環境中嵌入文本來模擬真實世界中的文本出現。專案使用了自訂的虛擬攝像機以及文本渲染技術來捕捉文本與其周圍環境的互動。

專案提供了一個包含 10K 圖像的數據集，這些圖像是從 30 個不同的場景中合成的。在 ICDAR 2015、ICDAR 2013 和 MLT 競賽的基準測試中，使用 SynthText3D 合成的數據顯示出良好的性能。

## UnrealText

- [**UnrealText**](https://github.com/Jyouhou/UnrealText/)

UnrealText 是一個使用 3D 圖形引擎來合成場景文本圖像的創新專案。此專案緊密結合了相關的學術論文，具體詳見論文[**《UnrealText: Synthesizing Realistic Scene Text Images from the Unreal World》**](https://arxiv.org/abs/2003.10608)。

UnrealText 利用 Unreal Engine，這是一款強大的遊戲開發工具，來創建極為逼真的文本場景。專案不僅提供了源代碼和使用說明，還包括了一個示範用的 UE 項目和 30 個預編譯場景可執行檔。此外，還發布了大規模的拉丁語/英語以及多語言合成場景文本數據集。

## SynthTIGER

- [**SynthTIGER**](https://github.com/clovaai/synthtiger)

SynthTIGER (Synthetic Text Image Generator) 是由 NAVER Corp. 開發的一個開源專案，旨在生成合成文本圖像以支援光學字元辨識（OCR）模型的訓練。該工具特別設計來改善文本辨識模型的表現，並於 2021 年國際文件分析和辨識會議（ICDAR）上發表。

SynthTIGER 支援多語言和多樣化的文本樣式生成，能夠產生豐富和多變的訓練數據。包括字體定制、顏色映射定制和模板定制，使用者可以根據具體需求調整生成過程。同時也提供了分割成多個小文件的大型數據集，方便下載和使用。

## TextRenderer

- [**text_renderer**](https://github.com/Sanster/text_renderer)

Text Renderer 是由 oh-my-ocr 組織開發的開源項目，用於產生用於深度學習 OCR 模型訓練的文字行圖像。 該專案特別注重模組化設計，使用者可以輕鬆添加不同的元件，如語料庫、效果和佈局。

Text Renderer 整合了 imgaug 來支援影像增強，支援在影像上渲染多種語料庫並套用不同效果，佈局負責多個語料庫之間的佈局。 此外，Text Renderer 支援產生垂直文本，並可產生與 PaddleOCR 相容的 lmdb 資料集。

## TextRecognitionDataGenerator

- [**TextRecognitionDataGenerator**](https://github.com/Belval/TextRecognitionDataGenerator)

TextRecognitionDataGenerator (TRDG) 是由 Belval 開發的一個開源項目，專門用於產生用於訓練光學字元辨識（OCR）軟體的文字影像樣本。 此計畫支援包括非拉丁文本在內的多語言文字生成，能夠透過不同的字體、背景和文字效果（如傾斜、扭曲、模糊等）來生成圖像。 TRDG 以其模組化和易於擴展的特性，成為了 OCR 模型開發者和研究人員的重要工具。

TRDG 可以透過命令列介面（CLI）使用，也提供了 Python 模組介面，使其能夠直接整合到訓練管道中。使用者可以透過簡單的命令產生包含隨機文字的圖像，或指定特定的文字內容。此外，TRDG 支援產生垂直文字和字元級掩碼，增加了生成文字影像的靈活性和適用性。
