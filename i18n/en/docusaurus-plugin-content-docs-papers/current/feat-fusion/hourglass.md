---
sidebar_position: 3
---

# Hourglass

## The Forgotten Elder

**[Stacked Hourglass Networks for Human Pose Estimation (2016.03)](https://arxiv.org/abs/1603.06937)**

---

:::info
The following content is compiled by ChatGPT-4, with manual proofreading, editing, and additional explanations.
:::

---

If you're someone who often navigates between different papers, the Hourglass architecture might not be unfamiliar to you. Let's quickly introduce this architecture:

The Hourglass architecture is designed specifically for human pose estimation tasks. It aims to comprehensively process and integrate features of all scales, striving to best capture various spatial relationships related to the human body. The research team demonstrated that by applying bottom-up and top-down processing methods repeatedly and combining intermediate supervised training, the network's performance can be greatly improved. Therefore, they named this new architecture the "stacked hourglass" network. The network is based on consecutive pooling and upsampling steps to generate the final prediction results.

This paper was proposed more than half a year earlier than FPN, but why is there such a huge difference in their fame? The most intuitive feeling is the citation count of the papers: as of August 2023, FPN has 20,816 citations, while Hourglass has 5,365. Why such a fourfold difference in citation count for architectures that seem similar?

- **Wait a second, did you say they're similar?**

## Model Architecture

We've discussed FPN before, and its architecture is probably familiar to you. Now, let's delve into the Hourglass architecture:

![hourglass_1](./resources/hourglass_1.jpg)

At first glance, they seem like entirely different architectures. However, let's follow the narrative of the paper and go through this model:

1. **Capturing Information at Different Scales**

    Imagine you're looking at a picture of a person's full body. To accurately understand the pose of this person, we need to simultaneously focus on the entire body and the details, such as the face and hands. However, this information may exist at different scales. The purpose of the Hourglass design is to capture information at these different scales simultaneously.

2. **Design of the Hourglass Structure**

    The structure of this model is like an hourglass. It consists of a series of convolutional layers (for feature extraction) and max-pooling layers (to reduce the resolution of the image). This allows retaining spatial information at each resolution without losing details.

3. **Upsampling and Feature Fusion**

    After the network processes lower resolutions, it starts upsampling, akin to zooming in on the image. Simultaneously, it combines features at different scales to integrate information about the entire body and details.

4. **Final Prediction**

    Finally, the network generates the final prediction through some processing. This prediction is a set of heatmaps, where different colors at different positions represent different features. Here, the network tries to predict the presence of human joints in the image, such as elbows and knees.

The fourth part belongs to the specific application domain, which we'll ignore for now.

Let's redraw the model architecture:

![hourglass_2](./resources/hourglass_2.jpg)

The green-boxed area belongs to the Backbone category, performing N stages of downsampling. The paper mentions that downsampling here is achieved through a series of convolutional layers coupled with max-pooling operations. Then, it enters the upsampling process, gradually increasing the resolution of feature maps and summing them up. Channel alignment is achieved through 1×1 convolutions.

Seeing this, isn't it obvious?

In terms of feature fusion, Hourglass is just FPN, and FPN is just Hourglass!

Of course, their application scenarios are different. Hourglass actually only takes the highest resolution layer of feature maps (P1) and repeats stacking multiple layers to extract keypoints effectively. FPN, on the other hand, doesn't specifically mention stacking many layers (although it's feasible in practice) but focuses on using feature maps of different resolutions (P3, P4, P5) to achieve multi-scale object detection.

So why such a huge difference in citation count between these two papers?

Actually, we can only give a speculative answer. One reason could be the difference in application scenarios—object detection is more popular, that's one. Another possible reason is the narrative structure of the papers: in the Hourglass paper, the emphasis is mainly on "stacking many layers" and "intermediate supervision."

## Hourglass's Origins?

Carefully going through the Hourglass paper, we can find an even earlier reference, which has a very similar structure:

**[Bottom-Up and Top-Down Reasoning with Hierarchical Rectified Gaussians (2015.07)](https://arxiv.org/abs/1507.05699)**

In this paper, the structure is not called "Hourglass"; it just mentions a "bottom-up" and "top-down" structure. Here's an excerpt from the paper:

> The main purpose of this paper is to explore a "bidirectional" structure that considers both top-down feedback: neurons influenced by neurons from higher and lower layers. The paper operates on neurons as having rectified latent variables within a quadratic energy function, which can be seen as a hierarchical rectified Gaussian model (RGs). The authors demonstrate that RGs can be optimized through quadratic programming (QP), which can be optimized through a recurrent neural network with rectified linear units. This enables RGs to be trained using GPU-optimized gradient descent.

Wait, what is he saying?

In simpler terms:

This paper explores a new approach to handle neurons in a more refined way, as if they were tuned to better suit specific tasks. This approach is applied in a mathematical model capable of better handling data like images. The researchers demonstrate that this approach can optimize through solving a specific mathematical problem, which can be solved using a special recurrent neural network. Such a structure enables us to utilize hardware resources more efficiently during computations.

If you still don't understand, that's okay. It's not the focus of this article. Remember, we're just trying to find the earliest literature that proposed this architecture. Below is the model architecture they used:

![hourglass_3](./resources/hourglass_3.jpg)

Seeing this familiar structure, it's reasonable to assume that it might be the inspiration for Hourglass. But is this the earliest design?

Actually, it might not be.

Because half a year earlier than this paper —

- **[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)**

— had already been proposed. U-Net's multiscale connectivity structure actually looks like this, with the only difference being the use of concatenation in U-Net. Unfortunately, this paper didn't cite U-Net, so we can't see the authors' summary and evaluation of this prior work.

## Conclusion

Although Hourglass and FPN are two architectures applied in different fields, their essence in feature fusion networks is actually the same—just different ways of using the same architecture.

On the same foundation, different application workflows are developed to solve different problems. FPN handles information at different scales through feature pyramids, mainly used for tasks like object detection and segmentation. In comparison, the Hourglass network extracts features at different detail levels through a hierarchical structure, particularly suitable for dense prediction tasks such as pose estimation.

If you've ever been confused about the essence of these two architectures, I hope this article provides some clarity.