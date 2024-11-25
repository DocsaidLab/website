---
sidebar_position: 10
---

# 参考文献

このテーマに関する論文は少ないですが、いくつか代表的な論文をリストアップし、研究の基礎資料として使用しています。

## 総合比較

|      モデル名      |  bg01  |  bg02  |  bg03  |  bg04  |  bg05  | 総合評価 |
| :----------------: | :----: | :----: | :----: | :----: | :----: | :------: |
|  HU-PageScan [1]   |   -    |   -    |   -    |   -    |   -    |  0.9923  |
| Advanced Hough [2] | 0.9886 | 0.9858 | 0.9896 | 0.9806 |   -    |  0.9866  |
|     LDRNet [4]     | 0.9877 | 0.9838 | 0.9862 | 0.9802 | 0.9858 |  0.9849  |
| Coarse-to-Fine [3] | 0.9876 | 0.9839 | 0.9830 | 0.9843 | 0.9614 |  0.9823  |
|  SEECS-NUST-2 [3]  | 0.9832 | 0.9724 | 0.9830 | 0.9695 | 0.9478 |  0.9743  |
|      LDRE [5]      | 0.9869 | 0.9775 | 0.9889 | 0.9837 | 0.8613 |  0.9716  |
|  SmartEngines [5]  | 0.9885 | 0.9833 | 0.9897 | 0.9785 | 0.6884 |  0.9548  |
|    NetEase [5]     | 0.9624 | 0.9552 | 0.9621 | 0.9511 | 0.2218 |  0.8820  |
|   RPPDI-UPE [5]    | 0.8274 | 0.9104 | 0.9697 | 0.3649 | 0.2162 |  0.7408  |
|   SEECS-NUST [5]   | 0.8875 | 0.8264 | 0.7832 | 0.7811 | 0.0113 |  0.7393  |

## 論文リスト

1. **HU-PageScan** はピクセル分類に基づいたカットモデルであり、効果は良好ですが、モデルのサイズと計算量が大きく、またモデルアーキテクチャに制限があるため、一部の遮蔽パターンに対する耐性が低いです。たとえば、指で角を掴んでいる状況では、実務的な要求を満たせません。
   - **Paper**: [HU-PageScan: a fully convolutional neural network for document page crop](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-ipr.2020.0532)
   - **Year.Month**: 2021.02
   - **Github**: [HU-PageScan](https://github.com/ricardobnjunior/HU-PageScan)

---

2. **Advanced Hough** は CV ベースのモデルであり、効果は良いものの、CV ベースのモデルは光量や角度に対して敏感であるといった欠点があります。
   - **Paper**: [Advanced Hough-based method for on-device document localization](https://www.computeroptics.ru/KO/PDF/KO45-5/450509.pdf)
   - **Year.Month**: 2021.06
   - **Github**: [hough_document_localization](https://github.com/SmartEngines/hough_document_localization)

---

3. **Coarse-to-Fine** と **SEECS-NUST-2** は深層学習に基づいたモデルで、再帰的最適化戦略を採用しています。効果は良好ですが、非常に遅いです。
   - Paper: [Real-time Document Localization in Natural Images by Recursive Application of a CNN](https://khurramjaved.com/RecursiveCNN.pdf) (2017.11)
   - **Paper**: [Coarse-to-fine document localization in natural scene image with regional attention and recursive corner refinement](https://sci-hub.et-fine.com/10.1007/s10032-019-00341-0)
   - **Year.Month**: 2019.07
   - **Github**: [Recursive-CNNs](https://github.com/KhurramJaved96/Recursive-CNNs)

---

4. **LDRNet** は深層学習に基づいたモデルで、提供されたモデルを使用してテストを行いましたが、このモデルは SmartDoc 2015 データセットに完全に適合しており、他のシーンに対しては全く一般化能力がありません。他のデータを追加して再訓練を試みましたが、最終的な性能もあまり良くなく、恐らく特徴融合の能力が不足しているためだと思われます。
   - **Paper**: [LDRNet: Enabling Real-time Document Localization on Mobile Devices](https://arxiv.org/abs/2206.02136)
   - **Year.Month**: 2022.06
   - **Github**: [LDRNet](https://github.com/niuwagege/LDRNet)

---

5. **LDRE**、**SmartEngines**、**NetEase**、**RPPDI-UPE**、**SEECS-NUST** 以下のモデルはすべて CV ベースのモデルです。
   - **Paper**: [ICDAR2015 Competition on Smartphone Document Capture and OCR (SmartDoc)](https://marcalr.github.io/pdfs/ICDAR15e.pdf)
   - **Year.Month**: 2015.11
   - **Github**: [smartdoc15-ch1-dataset](https://github.com/jchazalon/smartdoc15-ch1-dataset)
