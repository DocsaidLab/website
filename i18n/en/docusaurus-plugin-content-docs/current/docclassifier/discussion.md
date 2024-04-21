---
sidebar_position: 7
---

# Discussion

Based on our experiments, we have developed a model with promising performance. This model achieved over 90% accuracy on our test set and has demonstrated good results in practical applications.

Here, we will discuss some of our insights and experiences gained during the training process.

---

1. When considering which Margin loss to use, such as CosFace or ArcFace, it is crucial to pair it with [PartialFC](https://arxiv.org/abs/2203.15565). This significantly accelerates training speed, stabilizes convergence, and improves performance.

2. Regarding the variety of categories in text-image classification, we initially started with around 500 categories, then expanded to 800, 10,000, and finally incorporated an indoor dataset to increase the overall classification categories to approximately 400,000. This conclusion aligns with face recognition tasks: the effectiveness of the model is closely related to the diversity of the training data. Therefore, utilizing a large and diverse dataset ensures that the model learns sufficient features and can effectively differentiate between various categories.

3. Experimentation has shown that adopting **low-precision training** enhances the model's generalization ability. We believe this is because models are prone to overfitting, and low-precision training helps alleviate this issue. Directly setting low-precision training on the `trainer` is not feasible due to some operators not supporting this setting on CUDA. Therefore, we employed the `torch.set_float32_matmul_precision('medium')` method to achieve low-precision training.

4. Through experimentation, it was observed that LayerNorm performs better than BatchNorm in text-image classification tasks. This may be attributed to text images (e.g., street signs, document images) typically containing highly variable features, such as different fonts, sizes, and background noise. LayerNorm, by independently normalizing each sample, helps the model handle these variations more effectively. Conversely, in face recognition, using BatchNorm assists the model in learning to differentiate subtle differences from highly similar facial images. This ensures that the model can effectively recognize facial features under various conditions (e.g., lighting, angles, facial expressions).

5. When individually using CosFace and ArcFace, ArcFace showed better performance. However, after incorporating PartialFC, the situation changed, and CosFace performed better.

6. Pretraining is essential; attempting training without pretraining resulted in very poor performance. This could be due to the insufficient diversity of the provided dataset, necessitating pretraining to help the model learn more features. We once again thank timm for providing models that helped us save a significant amount of time and effort.

7. During the concatenation of the Backbone and Head, utilizing `nn.Flatten` to gather all information and integrate it into the feature encoding layer yields the best results. However, the drawback is the substantial parameter overhead. In lightweight model scenarios, even a 1MB increase in model size is significant. Therefore, we explored two directions: using `nn.GlobalAvgPool2d` to gather all information and integrating it into the feature encoding layer with `nn.Linear`, and using `nn.Conv2d` to first reduce the channel dimension to 1/4, referred to as Squeeze, followed by `nn.Flatten` and `nn.Linear` to integrate it into the feature encoding layer. Through experimentation, employing the Squeeze strategy proved effective. This strategy not only significantly reduces model size but also maintains model performance.

8. Introducing CLIP features is a good strategy as it not only improves model performance but also enhances model generalization. The core of this strategy is using the CLIP model to distill knowledge into our model, thereby incorporating CLIP model features into our model. This strategy has shown very positive results as the CLIP model possesses rich knowledge that helps our model learn more robust feature representations.

9. Adding a BatchNorm layer after LayerNorm is the key to pushing our model past the 90% mark. We believe this is because the characteristics of LayerNorm and BatchNorm complement each other. LayerNorm helps the model learn more stable features, while BatchNorm aids in learning cross-sample feature representations.
