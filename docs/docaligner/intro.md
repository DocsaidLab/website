---
sidebar_position: 1
---

# 介紹

本專案的核心功能稱為「**文件定位（Document Localization）**」。

- [**DocAligner Github**](https://github.com/DocsaidLab/DocAligner)

![title](./resources/title.jpg)

## 概述

此模型專門設計來辨識圖像中的文件，並將其攤平，以便進行後續的文字辨識或其他處理。

這裡提供兩種不同的模型：「熱圖模型」和「點回歸模型」，各具特點和適用場景，這些將在後續章節中詳細介紹。

在技術層面，我們選擇了 PyTorch 作為訓練框架，並在推論時將模型轉換為 ONNX 格式，以便在不同平台上部署。此外，我們使用 ONNXRuntime 進行模型推論，這使得我們的模型能在 CPU 和 GPU 上高效運行。

我們的模型在性能上達到接近最先進（SoTA）水平，並在實際應用中展示了即時（Real-Time）的推論速度，使其能夠滿足大多數應用場景的需求。

:::info
在深度學習領域以外的領域，『Localization』通常指文件在地化的意思，例如將文件翻譯成不同語言。在深度學習領域，則指的是定位圖像中的文件，並將其攤平的過程。
:::

:::tip
**攤平**：將文件中的文字或圖像進行矯正，使其在平面上呈現。
:::


## 定義

我們遵循該領域的先行者的定義，將文件的座標點的：

- **起點定為左上角**
- **終點定為左下角**

並使用四個座標點來表示文件的位置，依序為：『左上 > 右上 > 右下 > 左下』。

:::danger
如果在你的理解中，這些點位是根據文件的旋轉方向所定義的結果，這是錯誤的。

雖然在可視化的結果中，我們根據不同的座標點位使用不同的顏色，但該顏色並不代表文件本身的方向。若文件是反向的，則座標點的顏色仍然是按照『左上 > 右上 > 右下 > 左下』的順序。
:::
