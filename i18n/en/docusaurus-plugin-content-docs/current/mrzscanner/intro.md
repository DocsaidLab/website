---
sidebar_position: 1
---

# Introduction

MRZ (Machine Readable Zone) refers to a specific area on travel documents such as passports, visas, and identity cards, where information can be quickly read by machines. MRZ is designed and generated according to the guidelines in ICAO Document 9303, to speed up border control and improve the accuracy of information processing.

:::info
Interested readers can refer to: [**Document 9303**](./reference.md#icao-9303)
:::

Many people may not be familiar with MRZ, but almost everyone has a passport with an MRZ block, which looks like this, with the red box indicating the MRZ area:

<div align="center">
<figure style={{"width": "60%"}}>
![mrz example](./resources/img1.jpg)
<figcaption>Image source: [**MIDV-2020 Synthetic Dataset**](http://l3i-share.univ-lr.fr/MIDV2020/midv2020.html)</figcaption>
</figure>
</div>

---

In addition to passports, some countries also have MRZ blocks on identity cards, driver’s licenses, visas, and other documents.

We can see that MRZ blocks have several obvious characteristics:

1. **Fixed Structure**: Different types of MRZs have different structures, and each field's meaning is fixed.
2. **Clean Text Area**: The MRZ background is solid, the text is black, and there is a defined gap between characters.
3. **Simple Classification**: MRZ text contains only numbers and uppercase letters, with just 37 possible characters.

:::info
MRZ structures differ based on the type of document. The main types are:

1. **TD1 (ID cards, etc.)**: Three lines with 30 characters per line, totaling 90 characters.
2. **TD2 (Passport cards, etc.)**: Two lines with 36 characters per line, totaling 72 characters.
3. **TD3 (Passports, etc.)**: Two lines with 44 characters per line, totaling 88 characters.
4. **MRVA (Visa Type A)**: Two lines with 44 characters per line, totaling 88 characters.
5. **MRVB (Visa Type B)**: Two lines with 36 characters per line, totaling 72 characters.
   :::

## Structure Overview

We refer to the popular MRZ parsing GitHub project [**Arg0s1080/mrz**](https://github.com/Arg0s1080/mrz) to explain the structure of MRZ:

![field distribution](./resources/Fields_Distribution.png)

---

From the above diagram, we can clearly understand the meaning of each MRZ block:

1. **Type**: Document type, such as passport, identity card, visa, etc.
2. **Country Code**: Issuing country code.
3. **Surname**: Surname.
4. **Given Names**: Given names.
5. **Document Number**: Document number.
6. **National**: Nationality.
7. **Date of Birth**: Date of birth.
8. **Date of Expiry**: Expiry date.
9. **Optional**: Custom fields.

## Text Recognition

Our focus in this project is "MRZ Text Recognition." This topic is somewhat niche, and there aren’t many research papers on it. But when you break down the problem, it's essentially an OCR task. A few OCR models can be fine-tuned, and the problem is solved.

But that’s wasteful! Too wasteful!

OCR models are typically designed to recognize a wide variety of text types, including numbers, uppercase letters, punctuation, etc. The prediction head might cover thousands of characters, making the model complex and requiring more computational resources.

If we applied such a model directly to MRZ recognition, it would not be very professional, right?

Therefore, we must redesign the model specifically for MRZ characteristics. This specialized model can perform the task more efficiently without having to handle unnecessary text types, saving computational resources and improving recognition speed and accuracy.

## Two-Stage Recognition

Since we are designing a specialized model, we can divide MRZ recognition into two stages:

1. **Region Localization**: Using a lightweight model to focus on locating the MRZ region in the image.
2. **Text Recognition**: Using a lightweight model to focus on recognizing the text within the MRZ region.

We spent half a month completing the MRZ localization model and another month completing the MRZ recognition model. The overall performance is quite good. On our custom test set (around 300 MRZ documents), the full-image accuracy reached 97.02%, and the ANLS was 99.97%.

:::tip
Full-image accuracy and ANLS are the metrics we use to evaluate the model’s performance. For more information about these metrics, refer to another section: [**Model Evaluation**](./benchmark.md)
:::

:::info
Since each MRZ region contains at least 72 characters and at most 90 characters, through experimentation, if we aim for a full-image accuracy of 95%, the ANLS value needs to be at least around 99.95%. This is not an easy task.
:::

The only downside to this approach is...

### It’s Boring

No matter how you look at it, we feel that finishing this task smoothly is just doing our job.

Since the client gave us the requirement, we simply followed the steps and completed it. After delivering the solution to the client, we tossed it into the corner and began thinking about new solutions.

> **If we don't use a two-stage recognition approach, then it has to be single-stage recognition!**

We must directly recognize the text within the MRZ region from the raw image.

## Single-Stage Recognition

Then we continued to spend three more months completing a single-stage MRZ recognition model. The results were somewhat unsatisfactory, with a full-image accuracy of only about 40% and an ANLS of just 97%.

Honestly, we spent more time than we initially expected, and it was a bit of a loss. This problem turned out to be more difficult than we imagined. Several times we thought, “Why not just stick with the two-stage solution? It’s boring, but at least it’s accurate! Why make things harder for ourselves?”

There were several difficulties with the single-stage model:

1. The size and scale of the full-image search are inconsistent.
2. The MRZ region has unpredictable rotations.
3. The need for fine-grained perception leads to a computational explosion.

Given these challenges, the model had to remain lightweight to meet the requirements of mobile applications. These factors made the model difficult to converge and resulted in subpar performance.

:::info
We will go into the technical details in the next chapters: [**Model Design**](./model_arch.md)
:::

In the end, despite the frustration during the development process, we persisted and completed the task. Since the money and time spent can’t be recovered, we decided to open-source this solution and share it with everyone.

We view the single-stage solution as a "milestone." Our vision for the complete solution is a more robust, more accurate model that can handle more application scenarios.

:::tip
We will continue to read more papers and improve the model's performance in the future.
:::

## Playground

We have placed this model on this webpage, so you can try it out in the playground.

- [**MRZScanner-Demo**](https://docsaid.org/en/playground/mrzscanner-demo)

:::tip
If you find any bugs while using it, please notify us privately to avoid malicious exploitation, and we will address them as soon as possible.
:::

## Conclusion

In this project, we have completed the following features:

1. Verified the validity of synthetic datasets.
2. Completed a two-stage solution for MRZ region localization and recognition.
3. Integrated MRZ localization and recognition into a single-stage recognition model.
4. Integrated all MRZ document formats and provided a unified parsing interface.

If you are interested in this topic, feel free to test it out yourself, and we look forward to your feedback.

We also welcome suggestions and are happy to engage in discussions with you.
