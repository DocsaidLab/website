---
sidebar_position: 8
---

# References

The literature on this topic is sparse; we've compiled some of the more representative papers to serve as foundational material for research.

## Comparative Overview

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

## List of Papers

1. **HU-PageScan** is a segmentation model based on pixel classification. While it performs well, the model size and computational requirements are significant, and it lacks resistance to partial occlusions, such as scenarios where fingers hold the document corners, failing to meet practical needs.
    - **Paper**: [HU-PageScan: a fully convolutional neural network for document page crop](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-ipr.2020.0532)
    - **Year.Month**: 2021.02
    - **Github**: [HU-PageScan](https://github.com/ricardobnjunior/HU-PageScan)

---

2. **Advanced Hough** is a CV-Based model that performs well, but like all CV-Based models, it has drawbacks, such as sensitivity to light and angles.
    - **Paper**: [Advanced Hough-based method for on-device document localization](https://www.computeroptics.ru/KO/PDF/KO45-5/450509.pdf)
    - **Year.Month**: 2021.06
    - **Github**:  [hough_document_localization](https://github.com/SmartEngines/hough_document_localization)

---

3. **Coarse-to-Fine** and **SEECS-NUST-2** are deep learning-based models that use a recursive optimization strategy. While effective, they are slow.
    - **Paper**: [Real-time Document Localization in Natural Images by Recursive Application of a CNN](https://khurramjaved.com/RecursiveCNN.pdf) (2017.11)
    - **Paper**: [Coarse-to-fine document localization in natural scene image with regional attention and recursive corner refinement](https://sci-hub.et-fine.com/10.1007/s10032-019-00341-0)
    - **Year.Month**: 2019.07
    - **Github**:  [Recursive-CNNs](https://github.com/KhurramJaved96/Recursive-CNNs)

---

4. **LDRNet** is a deep learning-based model that we tested using their provided model. We found that the model was entirely fitted on the SmartDoc 2015 dataset, showing no generalization ability to other scenarios. We also tried to incorporate other data for training, but the performance was still not ideal, possibly due to the architecture's insufficient feature fusion capability.
    - **Paper**: [LDRNet: Enabling Real-time Document Localization on Mobile Devices](https://arxiv.org/abs/2206.02136)
    - **Year.Month**: 2022.06
    - **Github**:  [LDRNet](https://github.com/niuwagege/LDRNet)

---

5. **LDRE**, **SmartEngines**, **NetEase**, **RPPDI-UPE**, **SEECS-NUST** are all CV-Based models.
    - **Paper**: [ICDAR2015 Competition on Smartphone Document Capture and OCR (SmartDoc)](https://marcalr.github.io/pdfs/ICDAR15e.pdf)
    - **Year.Month**: 2015.11
    - **Github**:  [smartdoc15-ch1-dataset](https://github.com/jchazalon/smartdoc15-ch1-dataset)
