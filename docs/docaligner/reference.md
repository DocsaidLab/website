---
sidebar_position: 8
---

# 參考文獻

這個主題的論文比較少，我們把一些比較有代表性的論文列出來，用來作為研究的基礎材料。

## 綜合比較

| Models | bg01 | bg02 | bg03 | bg04 | bg05 | Overall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| HU-PageScan [1] | - | - | - | - | - | 0.9923 |
| Advanced Hough [2] |  0.9886 |  0.9858 |  0.9896 |  0.9806 |  - |  0.9866 |
| LDRNet [4] | 0.9877 | 0.9838 | 0.9862 | 0.9802 | 0.9858 | 0.9849 |
| Coarse-to-Fine [3] |  0.9876 |  0.9839 |  0.9830 |  0.9843 |  0.9614 |  0.9823 |
| SEECS-NUST-2 [3] |  0.9832 |  0.9724 |  0.9830 |  0.9695 |  0.9478 |  0.9743 |
| LDRE [5] | 0.9869 | 0.9775 | 0.9889 | 0.9837 | 0.8613 | 0.9716 |
| SmartEngines [5] |  0.9885 |  0.9833 |  0.9897 |  0.9785 |  0.6884 |  0.9548 |
| NetEase [5] |  0.9624 |  0.9552 |  0.9621 |  0.9511 |  0.2218 |  0.8820 |
| RPPDI-UPE [5] |  0.8274 |  0.9104 |  0.9697 |  0.3649 |  0.2162 |  0.7408 |
| SEECS-NUST [5] |  0.8875 |  0.8264 |  0.7832 |  0.7811 |  0.0113 |  0.7393 |

## 論文列表

1. **HU-PageScan** 是一個基於像素分類的切割模型，雖然他的效果比較好，但模型尺寸及運算量較大，且受限於模型架構，對於部分遮蔽的樣態抵抗力較低，例如手指抓著邊角的這種情境，無法滿足實務上的需求。
    - **Paper**: [HU-PageScan: a fully convolutional neural network for document page crop](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-ipr.2020.0532)
    - **Year.Month**: 2021.02
    - **Github**: [HU-PageScan](https://github.com/ricardobnjunior/HU-PageScan)

---

2. **Advanced Hough** 是 CV-Based 的模型，雖然效果不錯，但是使用 CV-Based 的模型，都會有一些缺點，例如對於光線和角度的敏感度。
    - **Paper**: [Advanced Hough-based method for on-device document localization](https://www.computeroptics.ru/KO/PDF/KO45-5/450509.pdf)
    - **Year.Month**: 2021.06
    - **Github**:  [hough_document_localization](https://github.com/SmartEngines/hough_document_localization)

---

3. **Coarse-to-Fine** 和 **SEECS-NUST-2** 是一個基於深度學習的模型，採用了遞迴優化的策略，效果不錯，但是很慢。
    - Paper: [Real-time Document Localization in Natural Images by Recursive Application of a CNN](https://khurramjaved.com/RecursiveCNN.pdf) (2017.11)
    - **Paper**: [Coarse-to-fine document localization in natural scene image with regional attention and recursive corner refinement](https://sci-hub.et-fine.com/10.1007/s10032-019-00341-0)
    - **Year.Month**: 2019.07
    - **Github**:  [Recursive-CNNs](https://github.com/KhurramJaved96/Recursive-CNNs)

---

4. **LDRNet** 是一個基於深度學習的模型，我們有使用他們提供的模型進行測試，發現該模型完全擬合在 SmartDoc 2015 資料集上，對於其他場景完全沒有泛化能力。我們也試著加入其他資料進行訓練，最終的表現也不理想，可能是這個架構對與特徵融合的能力不足。
    - **Paper**: [LDRNet: Enabling Real-time Document Localization on Mobile Devices](https://arxiv.org/abs/2206.02136)
    - **Year.Month**: 2022.06
    - **Github**:  [LDRNet](https://github.com/niuwagege/LDRNet)

---

5. **LDRE**、**SmartEngines**、**NetEase**、**RPPDI-UPE**、**SEECS-NUST** 以下的模型都是基於 CV-Based 的模型。
    - **Paper**: [ICDAR2015 Competition on Smartphone Document Capture and OCR (SmartDoc)](https://marcalr.github.io/pdfs/ICDAR15e.pdf)
    - **Year.Month**: 2015.11
    - **Github**:  [smartdoc15-ch1-dataset](https://github.com/jchazalon/smartdoc15-ch1-dataset)
