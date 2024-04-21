---
sidebar_position: 1
---

# Introduction

The core functionality of this project is called "**Document Classification**".

- [**DocClassifier Github**](https://github.com/DocsaidLab/DocClassifier)

![title](./resources/title.jpg)

:::info
This project has a unique origin. It was conceived by an expert in facial recognition systems, who happens to be my friend: [**Jack, L.**](https://github.com/Jack-Lin-NTU). He knew about my website, so he completed the initial programming and feasibility verification. Then, he entrusted me with this idea to further develop and publish it here. Special thanks to him for his contribution.
:::

:::tip
Upon seeing this title, you might smirk and think, "Isn't it just a classification model?"

Yes, and no.

This time, we aim to create an atypical classification model. While its application scope may be limited, its intrinsic interest is quite high.

It might not be what you imagine. Please continue reading.
:::

## Overview

In past project experiences, the classification model can be considered one of the most common machine learning tasks.

There's nothing particularly difficult about classification models. First, we build a backbone, then map the final output to multiple specific categories, and finally evaluate the model's performance using several metrics such as accuracy, recall, F1-Score, and so on.

Although this sounds straightforward, in practical applications, we encounter some challenges. Let's take the topic of this project as an example:

### Class Definition

In any classification task, clearly and precisely defining categories is crucial. However, if the categories we define are highly similar, the model may struggle to differentiate between them.

- For example: Documents from Company A vs. Documents from Company B.

These two categories are both documents from different companies, and their differences may not be significant, making it challenging for the model to distinguish between them.

### Data Imbalance

In most scenarios, data collection is a challenging issue, especially when it involves sensitive data. In such cases, we may encounter data imbalance problems, which can lead to the model's insufficient predictive power for minority categories.

### Data Augmentation

In the industry, there is a plethora of documents, and we constantly want to add more document categories. However, each time we add a new category, the entire model needs to be retrained or fine-tuned. This incurs a high cost, including but not limited to: data collection, labeling, retraining, reevaluation, deployment, etc. All processes need to be repeated.

### Class Sub-labels

Customer demands can be wild.

Let's assume there's a client who initially defines a document type, let's call it Document A.

Then, the client wishes to provide more sub-labels for Document A, such as:

- Damaged Document A
- Glare Document A
- First-generation format of Document A
- Second-generation format of Document A
- ...

Ignoring the fact that every time a sub-label is added, the model needs to be rerun.

From the perspective of model engineering, if we treat these labels as independent categories, it's "unreasonable" because they are all based on Document A. If we treat these labels as a multi-class problem, it's also "unreasonable" because different sub-labels correspond to different main document formats.

:::tip
You might think next: If we can't solve the problem, let's solve the person who raised the problem.

- No!

This is a machine learning problem.
:::

## Metric Learning

Stepping out of the document classification topic, you'll realize that this problem is actually about **metric learning**.

The main goal of metric learning is to measure the similarity between samples by learning the optimal distance metric. In the traditional machine learning field, metric learning typically involves mapping data from the original feature space to a new feature space, where similar objects are closer, and dissimilar objects are farther away. This process is usually achieved by learning a distance function that better reflects the true similarity between samples.

If you've read the previous paragraph and still don't understand, to summarize in one sentence: **Metric learning is a method for learning similarity**.

### Application Scenarios

Metric learning is crucial in two well-known application scenarios:

- **Face Recognition**: As we mentioned earlier, the number of faces is constantly increasing, and we can't keep retraining the model. Therefore, using the framework of metric learning can help us learn a better distance function, thereby improving the accuracy of face recognition.

- **Recommendation Systems**: The goal of recommendation systems is to recommend products that users might be interested in based on their historical behavior. In this process, we need to measure the similarity between users to find similar user behaviors and recommend products accordingly.

In these applications, accurately measuring the similarity between two objects is crucial for improving system performance.

## Problem Solving

Although not every classification problem is suitable for elevating to the level of metric learning, in this project, metric learning serves as a weapon that can indeed help us overcome the obstacles mentioned above.

- **Obstacle 1: Class Definition**

    Our goal is to learn a better distance function that can help us distinguish similar categories more effectively. So, we no longer need to define categories. The objects we want to classify will ultimately become a registration data.

- **Obstacle 2: Class Data Imbalance**

    We no longer need to collect a large amount of data because our model no longer relies on a large number of samples. We only need one sample, which is our registration data. The rest can be trained using other training data.

- **Obstacle 3: Class Expansion**

    Expanding classes only requires registering new data, without the need to retrain the model. This design can significantly reduce the training cost.

- **Obstacle 4: Class Sub-labels**

    This problem can be well addressed within the framework of metric learning. We can treat sub-labels as new registration data, which will not affect the original model. The distance between sub-labels and main labels in the feature space may be very close, but not exactly the same, thus effectively distinguishing between these two categories.

---

We first introduced the framework of metric learning: [**PartialFC**](https://arxiv.org/abs/2203.15565), which combines techniques such as [**CosFace**](https://arxiv.org/abs/1801.09414) and [**ArcFace**](https://arxiv.org/abs/1801.07698), enabling precise classification without predefining a large number of categories.

Subsequently, in further experiments, we introduced the [**ImageNet-1K dataset**](https://www.image-net.org/) and the [**CLIP model**](https://arxiv.org/abs/2103.00020). We used the ImageNet-1K dataset as the base, treating each image as a category. Through this operation, the number of classification categories could be expanded to approximately 1.3 million, providing the model with richer visual variations and increasing data diversity.

In the benchmark comparison at TPR@FPR=1e-4, compared to the original baseline model, the effect was improved by approximately 4.1% (77.2%->81.3%). If we further introduce the CLIP model on top of the ImageNet-1K, performing knowledge distillation during training, the effect can be further improved by approximately 4.6% (81.3%->85.9%).

In the latest experiments, we attempted to combine BatchNorm and LayerNorm and achieved gratifying results. Based on the original CLIP distilled model, the effect at TPR@FPR=1e-4 was improved by approximately 4.4% (85.9%->90.3%).

## Conclusion

In testing, our model demonstrated over 90% accuracy under the condition of one in ten thousand (TPR@FPR=1e-4) error rate. Moreover, there's no need to retrain when adding new classification types.

In summary, we've essentially brought over the operational workflow of a facial recognition system!

During our development process, we often exclaimed, "Can we really do this?" As mentioned earlier, this project's first-generation architecture (first author) had achieved some results but was still unstable. By the time this project was published, it was already the third-generation model (second author), with overall improvements. It's considered a good result.

Compared to our previously released "conventional" projects, this project is full of fun.

Therefore, we decided to make the architecture and experimental results of this project public. This will also be our "only" project where we publicly release trained models. We hope this project can bring you some inspiration. If you can find new application scenarios from the design principles of this project, you're welcome to share them with us.