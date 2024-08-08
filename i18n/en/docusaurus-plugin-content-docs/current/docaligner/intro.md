---
sidebar_position: 1
---

# Introduction

The core functionality of this project is called "**Document Localization**."

- [**DocAligner Github**](https://github.com/DocsaidLab/DocAligner)

![title](./resources/title.jpg)

## Overview

This model is specifically designed to identify documents within images and flatten them for further text recognition or other processing.

We provide two different models: the "Heatmap Model" and the "Point Regression Model," each with its characteristics and suitable applications, which will be detailed in subsequent chapters.

Technically, we chose PyTorch as the training framework and converted the model to ONNX format for inference, enabling deployment across various platforms. Moreover, we utilize ONNXRuntime for model inference, allowing our model to run efficiently on both CPU and GPU.

Our model achieves performance close to state-of-the-art (SoTA) levels and demonstrates real-time inference speed in practical applications, meeting the needs of most usage scenarios.

:::info
Outside the field of deep learning, "Localization" typically refers to the localization of documents, such as translating them into different languages. In the context of deep learning, it refers to the process of locating a document within an image and flattening it.
:::

:::tip
**Flattening**: Correcting the images within a document to display them on a flat surface.
:::

## Definition

Following the definitions set by pioneers in the field, we define the document's coordinates as:

- **Starting point at the top left corner**
- **Ending point at the bottom left corner**

We use four coordinate points to represent the position of the document, in order: "Top Left > Top Right > Bottom Right > Bottom Left."

:::danger
If in your understanding, these points are defined based on the document's orientation, that is incorrect.

Although we use different colors for different coordinate points in visualization results, these colors do not represent the document's orientation itself. If the document is upside down, the color sequence of the coordinate points remains "Top Left > Top Right > Bottom Right > Bottom Left."
:::
