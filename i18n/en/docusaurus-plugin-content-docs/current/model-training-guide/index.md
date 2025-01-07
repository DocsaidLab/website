---
title: Model Training Guide
sidebar_position: 2
---

In project development, due to varying personal preferences and requirements of developers, the configuration of models often differs. To facilitate better collaboration, we have decided to maintain a common base module and derive model training methods that cater to different needs based on it.

The base module consists of two main parts:

1. [**Capybara**](https://github.com/DocsaidLab/Capybara): Includes ONNXRuntime and OpenCV, primarily used for inference, implementing basic image processing functions.
2. [**Chameleon**](https://github.com/DocsaidLab/Chameleon): Based on the PyTorch module, primarily used for deep learning model training.

As some developers prefer using the Pytorch-Lightning framework, while others are more familiar with Huggingface or other frameworks, there will be differences in the specific implementation of model training.

We encourage and support developers to choose the appropriate tools and methods according to their own needs and share their experiences and outcomes based on this.

In the future, we will invite more developers to share their best practices and insights, continuously improving our model training guide.

---

![title](./resources/title.webp)

---

```mdx-code-block
import DocCardList from '@theme/DocCardList';

<DocCardList />
```
