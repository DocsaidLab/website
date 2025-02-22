---
sidebar_position: 6
---

# Evaluation

We use two evaluation metrics to assess the performance of the model:

## Average Normalized Levenshtein Similarity (ANLS)

This metric measures the similarity between the predicted text and the true text.

It is based on the Levenshtein edit distance, which calculates the minimum number of editing operations (insertion, deletion, replacement) required to transform the predicted result into the true value. The distance is then normalized to a value between 0 and 1, where a higher value indicates that the prediction is closer to the true value.

For example:

- True text: `hello`
- Predicted text: `helo`

Levenshtein distance = 1 (missing one `l`), calculated as:

$$
\text{ANLS} = 1 - \frac{\text{Levenshtein Distance}}{\max(\text{len}(y_{\text{true}}), \text{len}(y_{\text{pred}}))}
$$

- The normalized similarity is calculated as 0.8, indicating a high similarity between the prediction and the true value.

This metric is especially useful in OCR scenarios, where partial errors are acceptable, and it serves as a good standard for evaluation.

## Accuracy

This is the commonly seen metric for determining the overall recognition accuracy of the model on an image. If all the text in the model's prediction is correct, the image is considered "correctly recognized"; otherwise, it is considered "incorrect."

Accuracy is expressed as a percentage, calculated by dividing the number of correctly recognized images by the total number of test images.

**Example**:

- Number of test images: 100
- Number of correctly predicted images: 85
- Accuracy = (85 / 100) × 100% = 85%

This metric is suitable for applications requiring high precision, such as form processing or identity recognition, as any incorrect character could affect the usability of the result.

:::tip
**Why not use "Character Accuracy"?**

Another common evaluation method is **Character Accuracy** or **Word Accuracy**, which measures the accuracy of individual characters or words in the predicted text compared to the true text. However, Character Accuracy may not be ideal in some application scenarios:

1. **Small errors lead to a very low score**:

   - True text: `hello world`
   - Predicted text: `helo world`

   Since `hello` is misrecognized as `helo`, even if only one letter is incorrect, the word is still considered wrong, resulting in a dramatic drop in character accuracy.

   ***

2. **Word misalignment causes significant impact**:

   - True text: `hello world`
   - Predicted text: `ello horld`

   In this case, the misalignment of the words causes both words to be considered incorrect, even if most of the letters are still correct, leading to a lower score.

   ***

3. **Cannot quantify partially correct predictions**:

   - True text: `123456`
   - Predicted text: `12345`

   In some scenarios, we care more about whether the model's output is "close" to the correct answer rather than being identical.

   If we use Character Accuracy, the word `123456` would be considered incorrect, leading to a very low score. However, in many cases, this output is still acceptable.

   ***

Therefore, in our project, we choose not to use Character Accuracy and instead use Accuracy along with ANLS as evaluation metrics.
:::

## Evaluation Dataset

Evaluating the model in this project faces significant challenges.

First, there is no standard dataset available because it involves personal private data, and no one can openly provide such a dataset. As a result, we had to collect data ourselves and annotate it, but the dataset we collected lacks credibility and cannot be considered an authoritative standard:

> Would you trust someone who claims their model achieves 100% accuracy by collecting their own data?

Additionally, a well-known dataset in this field is **MIDV** (Mobile Identity Document Video dataset), which contains various machine-readable documents like passports and residence permits. However, it has its limitations:

1. **Insufficient data**: The **MIDV** dataset is small, and most of it is based on synthetic samples, which does not represent real-world application scenarios.
2. **Not focused on MRZ recognition**: **MIDV** includes various documents mainly for document localization, but it does not provide text localization and recognition annotations for MRZ.

Although MRZ text annotations are available in the **MIDV-2020** version, it still lacks localization area information and cannot be used as a complete evaluation dataset.

Currently, we have a small private test set, mostly from friends, containing about 300 passport and residence permit images. This test set is used for internal testing to help evaluate the model's performance. However, as mentioned earlier, due to privacy concerns, we cannot publicly release the full test data.

If you're interested, you can collect your own data and test it using our model.
