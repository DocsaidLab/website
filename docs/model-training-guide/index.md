---
title: 模型訓練指南
sidebar_position: 2
---

在專案開發中，由於開發者的個人習慣和需求各異，模型的配置方式往往有所不同。為了更好地協作，我們決定共同維護基底模組，並基於此衍生出適應不同需求的模型訓練方式。

基底模組分為兩個主要部分：

1. [**Capybara**](https://github.com/DocsaidLab/Capybara)：包含 ONNXRuntime 和 OpenCV，主要用於推論，實現基本的圖像處理功能。
2. [**Chameleon**](https://github.com/DocsaidLab/Chameleon)：基於 PyTorch 模組，主要用於深度學習模型的訓練。

由於部分開發者偏好使用 Pytorch-Lightning 框架，而另一些開發者則更熟悉 Huggingface 或其他框架，因此在模型訓練的具體實現上會存在差異。

我們鼓勵並支持開發者根據自身需求選擇合適的工具和方法，並在此基礎上分享經驗與成果。

未來，我們將邀請更多開發者分享他們的最佳實踐和心得，持續完善我們的模型訓練指南。

---

![title](./resources/title.webp)

---

```mdx-code-block
import DocCardList from '@theme/DocCardList';

<DocCardList />
```
