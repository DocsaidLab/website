---
sidebar_position: 7
---

# Discussion

Based on our experiments, we have developed a model that performs quite well.

Here, we will discuss some insights and experiences we gained during the training process.

---

- While our model can achieve scores close to SOTA, real-world scenarios are much more complex than this dataset. Therefore, we shouldn't overly focus on these scores. Our goal is simply to demonstrate the effectiveness of our model.

- In our experiments, we found that the current design of our model architecture does not perform well in zero-shot scenarios, meaning the model requires fine-tuning to achieve optimal results in new environments. In the future, we should explore more robust model architectures with better generalization capabilities.

- As mentioned in the Model Design section, we cannot directly address the challenge of amplification error. Therefore, the stability of the "Heatmap Regression Model" far exceeds that of the "Point Regression Model".

- We defaulted to using `FastViT_SA24` as the backbone for the heatmap model due to its effectiveness and computational efficiency.

- Through experimentation, we found that a 3-layer `BiFPN` outperforms a 6-layer `FPN`, so we recommend using `BiFPN` as the configuration for the Neck section. However, our implementation of `BiFPN` involves `einsum` operations, which may pose challenges for other inference frameworks. Therefore, if you encounter conversion errors when using `BiFPN`, consider switching to the `FPN` model.

- Although the "Heatmap Regression Model" demonstrates stability, it requires supervision on high-resolution feature maps, resulting in significantly higher computational costs compared to the "Point Regression Model".

- However, we cannot overlook the advantages of the "Point Regression Model", including its ability to predict corners beyond the document boundary, lower computational requirements, and a faster and simpler post-processing pipeline. Therefore, we will continue to explore and optimize the "Point Regression Model" to improve its performance.
