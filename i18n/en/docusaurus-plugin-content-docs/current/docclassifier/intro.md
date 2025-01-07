---
sidebar_position: 1
---

# Introduction

In past project experiences, classification models have been some of the most common machine learning tasks.

Classification models are not difficult to implement. First, we build a backbone, then map the final output to multiple specific categories. Finally, we evaluate the model’s performance using several metrics, such as accuracy, recall, F1-score, and so on.

Although this may sound straightforward, in practical applications, we often encounter some issues. Let’s take the topic of this project as an example:

- **Category Definition**

  In classification tasks, if the categories we define are highly similar, the model may have difficulty distinguishing between them. For example, “Company A insurance document” and “Company B insurance document.” Both categories belong to documents from a company, and their differences may be minimal, making it hard for the model to distinguish between the two.

- **Data Imbalance**

  In most scenarios, data collection can be the most challenging problem, especially when dealing with documents containing personal information. Data imbalance further leads to poor prediction performance for minority categories.

- **Data Augmentation**

  In our daily life, we are surrounded by a large number of documents, and we constantly want to add more document categories.

  However, every time we add a new category, the entire model needs to be retrained or fine-tuned, which incurs high costs, including data collection, labeling, retraining, reevaluation, deployment, and so on. All of these processes must be repeated.

- **Sub-labels for Categories**

  Customers' needs are often unpredictable.

  Let’s assume there’s a customer who first defines a document type, let’s call it Document A. Then, the customer wants to add more sub-labels for Document A, such as:

  - Damaged Document A
  - Glare Document A
  - First-generation format Document A
  - Second-generation format Document A
  - ...

  Let’s not even discuss the fact that adding each sub-label would require retraining the model.

  From the perspective of model engineering, it’s “irrational” to treat these labels as independent categories since they all belong to Document A. Likewise, it’s “unreasonable” to treat them as a multi-class problem because sub-labels corresponding to different document formats may vary.

:::tip
So you might think: Since we can’t solve the problem, let’s solve the person who raised it!

> No, you can’t!

This is a machine learning problem.
:::

## Metric Learning

Stepping out of the document classification problem, you’ll realize that what we’re really discussing here is **Metric Learning**.

The primary goal of metric learning is to learn the optimal distance measure to evaluate the similarity between samples. In traditional machine learning, metric learning typically involves mapping data from the original feature space to a new feature space, where similar objects are closer together and dissimilar objects are farther apart. This is usually achieved by learning a distance function that better reflects the true similarity between samples.

To summarize in one sentence: **Metric learning is a method for learning similarity**.

### Application Scenarios

A well-known application of metric learning is **Face Recognition**.

As mentioned earlier, the number of faces continues to grow, and we can’t always retrain the model. Therefore, by using a metric learning framework, we can learn a better distance function to improve the accuracy of face recognition.

## Solving the Problem

Although not every classification problem is suited to elevate to the level of metric learning, in this project, the weapon of metric learning can indeed help us overcome the obstacles mentioned earlier.

- **Obstacle 1: Category Definition**

  The goal of our learning is to learn a better distance function, which helps us better distinguish similar categories. Therefore, we no longer need to define categories. The objects we want to classify will ultimately only become registered data.

- **Obstacle 2: Data Imbalance**

  We no longer need to collect massive amounts of data because our model doesn’t rely on large samples. We only need one sample, which serves as our registered data, and other parts can be trained using different training data.

- **Obstacle 3: Category Expansion**

  Expanding categories only requires registering new data, without the need to retrain the model. This design greatly reduces the training cost.

- **Obstacle 4: Sub-labels for Categories**

  This issue can be well addressed within the metric learning framework. We can treat sub-labels as new registered data, which will not affect the original model. The distance between sub-labels and the main label in the feature space may be close, but not identical, thus effectively distinguishing these two categories.

---

We initially introduced the metric learning framework: [**PartialFC**](https://arxiv.org/abs/2203.15565), which combines technologies like [**CosFace**](https://arxiv.org/abs/1801.09414) and [**ArcFace**](https://arxiv.org/abs/1801.07698), enabling precise classification without the need for pre-defined categories.

Next, in further experiments, we introduced the [**ImageNet-1K dataset**](https://www.image-net.org/) and [**CLIP model**](https://arxiv.org/abs/2103.00020). We used ImageNet-1K as the base, treating each image as a category. This operation expanded the number of categories to about 1.3 million, providing the model with richer visual variations and increasing data diversity.

In the TPR@FPR=1e-4 benchmark, the performance improved by about 4.1% (77.2% -> 81.3%) compared to the baseline model. By introducing the CLIP model on top of ImageNet-1K, and conducting knowledge distillation during training, the performance improved by an additional 4.6% (81.3% -> 85.9%) in the same benchmark.

In the latest experiments, we combined BatchNorm and LayerNorm, achieving encouraging results. On top of the CLIP distillation model, the TPR@FPR=1e-4 performance improved by about 4.4% (85.9% -> 90.3%).

## Why Not Contrastive Learning?

Contrastive Learning and Metric Learning are both methods for learning the similarity between samples.

So why did we choose not to use contrastive learning this time?

It’s not because it’s not good; we just believe that at this stage, metric learning is a better fit.

### Benefits of Contrastive Learning

The greatest advantage of contrastive learning is its ability to handle unlabeled data well. For scenarios where data labeling is difficult or the dataset is enormous, it’s essentially a “lifesaver.”

Moreover, it excels at learning general features, which can be applied not only to classification tasks but also across tasks, such as object detection, semantic segmentation, and so on.

### But There Are Drawbacks

First, contrastive learning heavily relies on the design of negative samples. If the selection of negative samples is poor, either too simple or too complex, the model’s training performance may suffer.

Additionally, contrastive learning has high resource requirements because it needs a large number of negative samples to help the model understand “what is different,” leading to high computational costs. This is especially true when large training batches are required to provide enough negative samples, posing a challenge for our hardware resources.

Furthermore, contrastive learning is limited by its self-supervised design for unlabeled data, so it’s difficult for the model to learn highly precise features (such as an error rate of one in ten thousand). This is evident in the leaderboard of face recognition, where metric learning methods still dominate.

---

In conclusion, we chose “metric learning” to solve the problem. In the future, we’ll allocate time to explore the application of contrastive learning and may even combine the strengths of both methods, allowing the model to learn general features while possessing powerful similarity judgment abilities.

## Final Thoughts

In testing, our model demonstrated over 90% accuracy at a TPR@FPR=1e-4 error rate, with no need to retrain when adding new category types.

In simple terms, we’ve essentially transferred the operation process from a face recognition system over to this!

During the development, we often jokingly asked ourselves, “Can this really work?” As mentioned earlier, the first-generation framework (first author) had some effect but was still unstable. By the time this project was released, it was already the third-generation model (second author), and the overall performance showed significant improvement, making it a solid result.

Compared to our previous “standard” projects, this one is full of fun.

Therefore, we’ve decided to release the project’s framework and experimental results, hoping it will inspire you. If you also find a new application scenario from the design concepts of this project, feel free to share it with us.
