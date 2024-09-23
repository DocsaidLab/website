---
sidebar_position: 1
---

# Introduction

The core functionality of this project is called "**Document Localization**."

- [**DocAligner Github**](https://github.com/DocsaidLab/DocAligner)

![title](./resources/title.jpg)

## Overview

This task is essentially a "precursor" to OCR tasks.

In recent years, various general-purpose OCR models have become highly sophisticated. They can find all the text in unconstrained environments and provide recognition results without any pre-processing.

However, it comes with a cost—an expensive one.

---

This "expensive" aspect can be viewed from several angles:

### Inference Cost

In real-world scenarios, document recognition tasks take up a significant portion of the OCR field because people frequently deal with documents in their daily lives. In these activities, you're usually on the "receiving" end rather than the "active" side of the equation:

1. Don’t you need to show identification when opening a bank account?
2. Don’t you need to show your passport when traveling abroad?
3. Don’t you need to provide proof documents when applying for a visa?
4. Don’t you need to show contracts when purchasing insurance?

Although these tasks are common, that doesn’t mean it’s a profitable business, as there are numerous providers offering similar services. Affordable, high-quality products are everywhere.

If you use a general-purpose Optical Character Recognition (OCR) model (often a large language model, or LLM) to address this issue, the inference cost alone could put you in financial trouble...

and possibly worse.

:::tip
We are excluding those who have achieved financial freedom here.
:::

### Text is Usually Dense

In documents, text is often densely packed. This means that if you don’t want to miss any information, you need to scan the document at a very high resolution.

For example, when we typically start with object detection, we often use a resolution of $640 \times 640$. However, in the document localization scenario, this resolution may be increased to $896 \times 896$, $1536 \times 1536$, or even higher.

If we don’t break down the model's functionality and instead push an LLM directly in a high-resolution environment, aside from the inference cost, do you know that buying a single V100 GPU for training now costs between \$30,000 ~ \$50,000 USD? LLMs are outrageously expensive!

### The Large Number of Chinese Characters

The classification of Chinese characters, which includes both simplified and traditional formats, covers over 100,000 unique characters. Compared to Latin-based scripts, the number of characters is three orders of magnitude larger.

The model must first identify the text in complex backgrounds and then extract key features from the intricate details of the characters, sometimes just a few pixels in size. This greatly increases the required number of parameters and computational load.

While we certainly aspire to an end-to-end solution that solves all problems with a single model, such models do exist now, and there will only be more in the future.

However, inference is expensive, and the return on investment is low. From any angle, it’s currently a losing business.

### Functional Decomposition

As a result, we need a more cost-effective approach, making model decomposition a necessary choice.

This is also the purpose of our project:

- **To accurately locate the document areas we are interested in within chaotic environments, and flatten them for subsequent text recognition or other processing.**

Once document localization is achieved, the next steps involve text localization, then text recognition, and finally semantic analysis and understanding.

This process might seem cumbersome, but when balancing cost and efficiency, it is a relatively optimal solution.

## Model Functionality

This model is specifically designed to recognize documents in images, precisely locating the four corners of the document. This allows the user to flatten the document for subsequent text recognition or other processing.

We offer two different models here: the "Heatmap Model" and the "Point Regression Model," each with its own strengths and applicable scenarios. These will be detailed in subsequent sections.

On the technical side, we chose PyTorch as the training framework and converted the model to ONNX format for inference to facilitate deployment on different platforms. Additionally, we use ONNXRuntime for model inference, allowing efficient execution on both CPUs and GPUs.

Our model achieves near state-of-the-art (SoTA) performance and demonstrates real-time inference speeds in practical applications, making it suitable for most use cases.

:::info
In fields outside deep learning, "Localization" usually refers to translating a document into different languages.

In the context of deep learning, however, it refers to the process of locating documents within images and flattening them.
:::

:::tip
**Flattening**: The process of projecting a skewed document in 3D space onto a 2D plane (e.g., through perspective transformation), so that it appears flat on the plane.
:::

## Definition

We follow the definitions established by pioneers in this field and use the following convention for document corner points:

- **The starting point is the top-left corner**
- **The ending point is the bottom-left corner**

Four corner points are used to represent the document’s position, listed in the order: 'Top-left > Top-right > Bottom-right > Bottom-left.'

:::danger
Although we use different colors for different corner points in the visualized results, these colors do not indicate the document's orientation.

In other words: **No matter how the text is oriented, the model will always define the top-left corner as the starting point and the bottom-left corner as the endpoint.**
:::

## Conclusion

Initially, we aimed to develop a zero-shot model capable of running smoothly on mobile devices. The goal was to create a model that could generalize to all types of documents worldwide, without needing any annotations or fine-tuning, and be ready for immediate use.

However, we later encountered limitations related to the model's size, making this goal seem somewhat out of reach.

This presented a difficult choice: increasing the model size would go against our original intention, but if we didn’t scale it up, we would have to change the architecture. Changing the architecture, however, could compromise the model's ability to generalize effectively.

After several months of hard work, we ultimately decided to compromise on the zero-shot goal, prioritizing **accuracy** as our top priority.

If you're interested in this topic, feel free to test the model yourself. We look forward to receiving your feedback.

We also welcome any suggestions and would be happy to engage in discussions with you.
