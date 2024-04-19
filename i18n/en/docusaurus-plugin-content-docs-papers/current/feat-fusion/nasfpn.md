---
sidebar_position: 4
---

# NAS-FPN

## Money Talks: NAS-FPN

**[NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection (2019.04)](https://arxiv.org/abs/1904.07392)**

---

:::info
The following content is compiled by ChatGPT-4, with manual proofreading, editing, and additional explanations.
:::

---

Since the advent of FPN, feature fusion has been a hotly debated topic. Here's a chronological list:

- 2017.01 -> [DSSD : Deconvolutional single shot detector](https://arxiv.org/abs/1701.06659)
- 2017.07 -> [RON: reverse connection with objectness prior networks for object detection](https://arxiv.org/abs/1707.01691)
- 2017.07 -> [Deep layer aggregation](https://arxiv.org/abs/1707.06484)
- 2017.09 -> [StairNet: top-down semantic aggregation for accurate one shot detection](https://arxiv.org/abs/1709.05788)
- 2017.11 -> [Single-shot refinement neural network for object detection](https://arxiv.org/abs/1711.06897)
- 2018.03 -> [Path Aggregation Network for Instance Segmentation](https://arxiv.org/abs/1803.01534) (PANet here)
- 2018.08 -> [Scale-transferrable object detection](https://ieeexplore.ieee.org/document/8578160)
- 2018.08 -> [Deep feature pyramid reconfiguration for object detection](https://arxiv.org/abs/1808.07993)
- 2018.10 -> [Parallel feature pyramid network for object detection](https://link.springer.com/chapter/10.1007/978-3-030-01228-1_15#chapter-info)

PANet is the most commonly heard of among these. Besides PANet, the aforementioned papers have hundreds to thousands of citations. It's recommended to give them a read when you have time.

So, which one should you choose?

Google wanted to know the answer too, which led to the publication of the paper NAS-FPN.

Can you guess the core concept? "I don't know which one's better, so let's use an algorithm…"

:::tip
Isn't that a bit off? But then again, it's so Google.

Remember the NasNet series? They're about searching network architectures. Eventually, they even came up with another paper called EfficientNet, which you might have heard of.

Besides network architecture, chip design can also use NAS. Now, using NAS for feature fusion is just another practical move.
:::

## What's NAS?

NAS stands for Neural Architecture Search, a crucial research direction in deep learning. Its main goal is to automatically find the best neural network architecture to solve specific tasks. Neural network architectures typically consist of multiple layers, neurons, and connections, and the design of these architectures can have a significant impact on the performance of the model.

Traditionally, neural network design has been mostly a manual process, requiring extensive experimentation and tuning by experts, which is time-consuming and requires domain expertise. NAS aims to simplify this process by automating it, allowing machines to explore and discover the best neural network architectures.

In NAS, a search space is defined, containing variants of all possible neural network architectures. Then, using different search strategies such as genetic algorithms, reinforcement learning, evolutionary algorithms, etc., the system automatically generates, evaluates, and selects these architectures to find the best one for a specific task.

Generally, the pros and cons of NAS are:

### Pros

- **Automation**: Can automatically explore and find the best neural network architecture, reducing the need for manual tuning and design work, thus saving time and resources.
- **Optimization**: Can find the best neural network structure for specific tasks and datasets, improving model performance and potentially surpassing manually designed models in some cases.
- **Flexibility**: Not limited to specific tasks or architectures, can adapt to different application scenarios, and generate models suitable for specific requirements.
- **Innovation**: Can lead to the discovery of new neural network structures, potentially bringing innovative model architectures and further advancing deep learning.

### Cons

- **Computational Resource Consumption**: Search process may require significant computational resources, including GPUs or TPUs, and a considerable amount of time, which may limit its practical application.
- **Complexity**: Size of the search space and the number of possible combinations may make the search process very complex, requiring more advanced algorithms and techniques for effective search.
- **High Dependency on Datasets**: Found best architectures may heavily depend on the specific dataset used for the search, and cannot guarantee superior performance on other datasets.
- **Stochasticity**: Search process may have some level of randomness, different search runs may yield different results, posing a challenge to the stability of the results.

## Actually, There Are More Cons

After reading about the pros and cons of NAS, you might be particularly interested in its flexibility and innovation. But the reality is that over 90% or more of practitioners lack the resources to build their own search systems, often having to rely on the results brought by this technology. This immediately leads to another question:

- **Does my use case match the paper's scenario?**

Here, the use case includes the feature distributions of inference data, training data, the search space for solving problems, etc. If there's a chance that the answer is no, then this optimized architecture might, perhaps, not be…

- **Suitable.**

So, why talk about this paper?

Firstly, we might be part of that 10% who are interested. This paper demonstrates how to design a search architecture and find the most suitable feature fusion method for one's own usage scenario. Secondly, it showcases some results obtained from automated searches, which can provide some inspiration for future designs.

## Problem Solving

### NAS-FPN Model Design

![nasfpn_1](./resources/nasfpn_1.jpg)

The primary goal of this study was to find a better FPN architecture. In the academic discourse, a model typically begins with a basic structure called the Backbone, which can be freely swapped, such as ResNet, MobileNet, etc.

Following the Backbone is the Neck, where FPN typically resides. Its main job is multiscale feature concatenation, which is the focus here.

It's worth mentioning that in this study, the authors used a framework called "RetinaNet" as the foundation. RetinaNet's backbone employs ResNet, while its neck employs FPN.

:::tip
The main theme of the RetinaNet paper is actually FocalLoss. The RetinaNet architecture inside it is a simple combination for applying FocalLoss.
:::

### Merging Cells

![nasfpn_2](./resources/nasfpn_2.jpg)

In NAS-FPN, a new concept called "Merging Cell" was proposed based on the original FPN design.

A Merging Cell is a small module responsible for "merging" two different input feature layers into a new output feature layer. This merging process consists of the following steps:

1. Select the first feature layer: Choose one from multiple candidate feature layers (could be C3, C4, C5, etc.), denoted as hi.
2. Select the second feature layer: Again, choose one from multiple candidate feature layers, denoted as hj.
3. Determine the size of the output feature: Choose a resolution size, which will be the size of the new merged feature layer.
4. Select the merge operation: Use a specific mathematical operation (such as addition or global pooling) to merge hi and hj.

In step 4, as shown in the diagram, two binary operations were designed: summation and global pooling. These operations were chosen because they are simple and efficient, requiring no additional trainable parameters.

If hi and hj have different sizes, upsampling or downsampling is applied to make them the same size before merging. The merged new feature layer undergoes a ReLU activation function, a 3×3 convolutional layer, and a BatchNorm layer to enhance its expressive capability. Thus, FPN can continuously merge and improve feature layers through multiple such Merging Cells, ultimately generating a set of better multiscale feature layers (P3, P4, P5, etc.).

## Discussion

Experimental data shows that with the increase of training steps, the controller is able to generate better and better subnetwork architectures. This process reaches a stable state after about 8000 training steps, meaning the number of unique architectures added begins to converge.

Finally, based on the optimization results of rewards, the authors selected the architecture with the highest AP for further training and evaluation.

This architecture was sampled during the first 8000 steps of training and sampled multiple times in subsequent experiments.

Subsequently, the authors demonstrated the FPN architecture obtained by the NAS algorithm as follows:

![nasfpn_5](./resources/nasfpn_5.jpg)

This diagram might look complex at first glance, but with annotations:

![nasfpn_3](./resources/nasfpn_3.jpg)

With annotations, we can now take a closer look at the results of NAS-FPN.

Firstly, in the initial FPN (a), it's not exactly FPN; it's a "FPN-like" structure because it outputs feature maps differently and has a different data flow sequence, though it's consistent with FPN in essence. However, the original FPN doesn’t have this many layers of convolutional layers.

Next, looking at the experimental results of NAS-FPN from (b) to (f), as AP scores keep improving, we can observe that the way of searching architectures ultimately verifies the design philosophy of the PANet paper, i.e., diagram (f):

- Data must be fused from top to bottom.
- Data must then be fused from bottom to top.
- Although the details might be slightly different, the essence is captured.

![nasfpn_4](./resources/nasfpn_4.jpg)

## Conclusion

In previous research, feature fusion architectures have mostly been derived through manual design and experimentation. The reliability and scale of this approach have always been questioned. Indeed, experimental research, while providing insights, often has its value limited by the scale and design of the experiments.

Perhaps we can accept that the "theoretical foundation" of certain conclusions might be insufficient and acknowledge that conclusions derived through "experimentation" suffice. But how do we convince others that the scale of these experiments is sufficient?

However, NAS-FPN offers a new perspective on this issue, with a precise search architecture and unprecedented computational scale (perhaps no other company has the resources or willingness to spend on such computations). This not only confirms the correctness of PANet's design philosophy but also reveals the potential inefficiencies in its connection method.

I believe this is the value of this paper. This method of combining NAS search results not only enhances the credibility of previous research but also provides new directions for future research.
