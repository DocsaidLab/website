---
sidebar_position: 5
---

# Evaluation

We utilized the [**SmartDoc 2015**](https://github.com/jchazalon/smartdoc15-ch1-dataset) dataset for our testing.

## Protocol

We employ the **Jaccard Index** as our measure, which summarizes how well different methods perform in correctly segmenting page contours and penalizes those that fail to detect document objects in certain frames.

The evaluation process starts by using the size and coordinates of the document in each frame to perform a perspective transform on the quadrilateral coordinates of the submitted method S and the ground truth G, obtaining the corrected quadrilaterals S0 and G0. This transformation ensures that all evaluation metrics are comparable within the document reference system. For each frame f, we calculate the Jaccard Index (JI), an indicator of the degree of overlap of the corrected quadrilaterals, defined as the intersection polygon of the detected quadrilateral and the ground truth quadrilateral divided by their union polygon. The overall score for each method is the average of the scores across all frames in the test dataset.

## Results

The following are the evaluation results of our models on the [**SmartDoc 2015**](https://github.com/jchazalon/smartdoc15-ch1-dataset) dataset:

| Models | bg01 | bg02 | bg03 | bg04 | bg05 | Overall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FastViT_SA24 |  0.9944 |  0.9932 |  0.9940 |  0.9937 |  0.9929 |  0.9937 |
| MBV2_140 |  0.9917 |  0.9901 |  0.9921 |  0.9899 |  0.9891 |  0.9909 |
| FastViT_T8 |  0.9920 |  0.9894 |  0.9918 |  0.9896 |  0.9888 |  0.9906 |
| LC100 |  0.9908 |  0.9877 |  0.9905 |  0.9894 |  0.9854 |  0.9892 |
| LC050 |  0.9847 |  0.9822 |  0.9865 |  0.9811 |  0.9722 |  0.9826 |
| PReg-LC050-XAtt |  0.9663 |  0.9606 |  0.9664 |  0.9630 |  0.9199 |  0.9596 |

## Analysis of Results

- Although our models achieve scores close to SoTA, real-world scenarios are much more complex than this dataset, so do not overemphasize these scores; we simply aim to prove our models are effective.

- Due to our efforts to reduce the size and computational demands of our models, we found that the models have poor zero-shot capability, meaning they need fine-tuning for best results in new environments.

- Experiments showed that "heatmap regression models" have far greater stability than "point regression models," so we continue to recommend using heatmap models.

- We default to using `FastViT_SA24` as the backbone for the heatmap model because of its effective performance and computational efficiency. You can replace it with a larger, more accurate backbone network if you wish.

- Through experimentation, `BiFPN` (3 layers) proved superior to `FPN` (6 layers), so we recommend using `BiFPN`. However, `BiFPN` involves `einsum` operations, which may cause issues with other inference frameworks, so if you encounter errors while using `BiFPN`, consider switching to an `FPN` model.

- Although "heatmap regression models" show stable performance, they require supervision on high-resolution feature maps, thus significantly increasing computational load compared to "point regression models."

- However, we still see the advantages of "point regression models," including but not limited to: the ability to predict corner points outside of the image frame; lower computational demands; and simple, fast post-processing steps. Therefore, we continue to optimize "point regression models" to enhance their performance.

## Parameter Settings

The table below details the parameter settings used for each model:

| Model Name   | ModelType | ModelCfg        |
|:------------:|:---------:|:---------------:|
| FastViT_SA24 | heatmap   | fastvit_sa24    |
| MBV2-140     | heatmap   | mobilenetv2_140 |
| FastViT_T8   | heatmap   | fastvit_t8      |
| LC100        | heatmap   | lcnet100        |
| LC050        | heatmap   | lcnet050        |
| PReg-LC050-XAtt | point  | lcnet050        |

For example, to use the LC050 model, call as follows:

```python
from docaligner import DocAligner

model = DocAligner(model_type='heatmap', model_cfg='lcnet050')
```

## Comparative Overview

The table below compares each model name based on parameters, FP32 size, FLOPs, and overall score:

| Model Name   | Parameters (M) | FP32 Size (MB) | FLOPs(G) | Overall Score |
|:------------:|:--------------:|:--------------:|:--------:|:-------------:|
| FastViT_SA24 |     20.8       |     83.1       |   8.5    |     0.9937    |
| MBV2-140     |      3.7       |     14.7       |   2.4    |     0.9909    |
| FastViT_T8   |      3.3       |     13.1       |   1.7    |     0.9906    |
| LC100        |      1.2       |      4.9       |   1.6    |     0.9892    |
| LC050        |      0.4       |      1.7       |   1.2    |     0.9826    |
| PReg-LC050-XAtt |   1.1       |      4.5       |   0.22   |     0.9596    |

:::tip
Choosing a model is a process of trade-offs; when you need a smaller model, `LC050` is a great option, though the overall score is lower; alternatively, you can use the default `FastViT_SA24`, but it will occupy more space.
:::