---
sidebar_position: 1
---

# Introduction

MRZ (Machine Readable Zone) refers to a specific area on travel documents such as passports, visas, and identity cards, where the information can be quickly read by machines. MRZ is designed and generated according to the International Civil Aviation Organization (ICAO) Document 9303, which helps speed up border checks and improve the accuracy of information processing.

:::info
Interested readers can refer to: [**Document 9303**](./reference.md#icao-9303)
:::

Most people may not know what MRZ is, but they usually have a passport in hand, which contains an MRZ block that looks like this, with the red-bordered part:

<figure>
![mrz example](./resources/img1.jpg)
<figcaption>Image source: [**MIDV-2020 Synthetic Dataset**](http://l3i-share.univ-lr.fr/MIDV2020/midv2020.html)</figcaption>
</figure>

---

In addition to passports, identity cards, driver's licenses, visas, and other documents in some countries also contain MRZ blocks.

We can observe that MRZ blocks have several distinct features:

1. **Fixed Structure**: Different types of MRZ have different structures, and the meaning of each field is fixed.
2. **Clean Text Area**: The MRZ background is monochrome, the text is black, and there is a consistent gap between characters.
3. **Simple Classification**: The MRZ block contains only numbers and uppercase letters, with only 37 possible characters in total.

:::info
The structure of MRZ varies depending on the document type, mainly including the following:

1. **TD1 (Identity card, etc.):** Composed of three lines with 30 characters per line, totaling 90 characters.
2. **TD2 (Passport cards, etc.):** Composed of two lines with 36 characters per line, totaling 72 characters.
3. **TD3 (Passports, etc.):** Composed of two lines with 44 characters per line, totaling 88 characters.
4. **MRVA (Visa Type A):** Composed of two lines with 44 characters per line, totaling 88 characters.
5. **MRVB (Visa Type B):** Composed of two lines with 36 characters per line, totaling 72 characters.
   :::

## Structure Overview

We refer to the well-known MRZ parsing project on GitHub, [**Arg0s1080/mrz**](https://github.com/Arg0s1080/mrz), to explain the MRZ structure:

![field distribution](./resources/Fields_Distribution.png)

---

From the above diagram, we can clearly understand the meaning of each MRZ block:

1. **Type**: Document type, including passport, ID card, visa, etc.
2. **Country Code**: Issuing country code
3. **Surname**: Surname
4. **Given Names**: Given names
5. **Document Number**: Document number
6. **National**: Nationality
7. **Date of Birth**: Date of birth
8. **Date of Expiry**: Expiry date
9. **Optional**: Custom fields

## Text Recognition

The topic of this article is "MRZ Text Recognition," which is a relatively niche subject with few research papers. However, if we break down the problem, it's essentially an OCR problem. By fine-tuning a few OCR models, the problem can be solved.

But that's a waste! A huge waste!

OCR models are generally designed to recognize various types of text, including numbers, uppercase and lowercase letters, punctuation marks, etc. The prediction head might cover thousands of text classes, making such models more complex and requiring more computing resources.

If such a model were directly applied to MRZ recognition, it would seem unprofessional, right?

Therefore, we need to redesign the model specifically for MRZ's characteristics. A dedicated model like this can more efficiently complete the task, avoiding unnecessary text types, thus saving computing resources and improving recognition speed and accuracy.

### Two-Stage Recognition

Since we are designing a dedicated model, we can divide MRZ recognition into two stages:

1. **Region Localization**: Use a lightweight model focused on locating the MRZ block in the image.
2. **Text Recognition**: Use a lightweight model focused on recognizing the text within the MRZ block in the image.

We did it! We spent one week building the MRZ localization model, and another week for the MRZ recognition model. Combined, the two models are about 5 MB, and the recognition accuracy per frame of image is about 95%.

:::tip
**Definition:** The accuracy mentioned above refers to the correct recognition of all text within the MRZ block. If even one character is incorrect, the entire image is considered incorrect.
:::

The only downside to this approach is...

### It's Boring

No matter how you look at it, we think that completing this task "smoothly" is just a procedural thing.

Since the customer sent the request, we just followed the steps and completed it. After delivering it to the customer, we threw the solution into the corner and began thinking about new solutions.

> **If we don't use two-stage recognition, then it must be single-stage recognition!**

We had to directly recognize the MRZ block's text from the raw image.

### Single-Stage Recognition

We then spent another three months to complete a single-stage MRZ recognition model.

To be honest, we spent more time than expected, and it was a bit of a loss. This problem was more difficult than we imagined. Several times we thought about giving up—after all, the two-stage solution was boring, but at least it was accurate! Why bother creating more trouble for ourselves?

The challenge with the single-stage model lies in that it must search the entire image for MRZ blocks of varying sizes and orientations and recognize the text within them. On top of that, the model must remain lightweight to meet mobile application requirements. These factors made model convergence difficult, and the results weren't great.

:::info
We will introduce the technical details in the upcoming chapter: [**Model Design**](./model_arch.md)
:::

In conclusion, while we were frustrated during development, we persevered and completed it. Since the time and money spent cannot be recovered, we decided to open-source this solution and share it with everyone.

We consider the entire single-stage solution as a "milestone." The ideal model in our minds should be more robust and accurate, and capable of handling more application scenarios.

:::tip
We will continue reading more papers and work on improving the model's performance in the future.
:::

## Model Evaluation

This part is really difficult.

First, there is no standard dataset for this type of problem, so we had to synthesize our dataset and annotate it ourselves, which doesn't hold much credibility:

> Otherwise, everyone would just collect their own data and claim to have 100% accuracy! Brilliant!

Secondly, while MIDV provides some data, most of it is based on synthetic samples. Fine-tuning models with it doesn't yield great results, let alone using it to evaluate model performance.

Therefore, in this project, we can't provide a complete model evaluation report like in previous projects.

## Final

In this project, we completed several functions:

1. Validated the effectiveness of the synthetic dataset.
2. Integrated MRZ localization and recognition to create a single-stage recognition model.
3. Integrated MRZ files of all formats and provided a unified parsing interface.

We also borrowed some real passports and residence permits from friends and tested them on passports from different countries: under certain regulated conditions, we achieved fairly stable recognition results.

If you are interested in this topic, feel free to test it yourself, and we look forward to your feedback.

Also, feel free to leave suggestions. We are happy to communicate with you.
