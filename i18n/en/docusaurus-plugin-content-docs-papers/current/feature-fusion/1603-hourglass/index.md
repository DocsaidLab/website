# [16.03] Hourglass

## The Forgotten Elder

**[Stacked Hourglass Networks for Human Pose Estimation](https://arxiv.org/abs/1603.06937)**

---

If you are a developer who frequently explores various papers, the Hourglass architecture may not be entirely unfamiliar to you. Let's quickly introduce this architecture:

The Hourglass architecture is designed specifically for human pose estimation tasks. By comprehensively processing and integrating features at all scales, it aims to optimally capture various spatial relationships related to the human body. The research team demonstrated that by repeatedly applying bottom-up and top-down processing and incorporating intermediate supervision, the network's performance could be significantly enhanced. Therefore, they named this new architecture the "Stacked Hourglass" network. This network, based on successive pooling and upsampling steps, is used to generate the final prediction.

This paper was published more than six months before the FPN paper, but why is there such a significant difference in their fame? The most intuitive indicator is the number of citations. As of August 2023, FPN has been cited 20,816 times, whereas Hourglass has been cited 5,365 times. How can two architectures with similar designs have such a vast difference in citations?

- **Wait, you said they are similar?**

## Model Architecture

We have discussed FPN before, and you are likely familiar with its architecture. Now, let's look at the Hourglass architecture:

![hourglass_1](./img/hourglass_1.jpg)

At first glance, they seem like entirely different architectures, but let's walk through the Hourglass model according to the paper's description:

1. **Capturing Information at Different Scales**

   Imagine observing an image with a person's full body. To accurately understand the person's pose, we need to focus on both the entire body and detailed parts like the face and hands. These pieces of information may be at different scales. The Hourglass design aims to capture these different scales simultaneously.

2. **Design of the Hourglass Structure**

   This model's structure resembles an hourglass. It consists of a series of convolutional layers (for feature extraction) and max-pooling layers (to reduce image resolution). This approach retains spatial information at each resolution, preventing the loss of details.

3. **Upsampling and Feature Combination**

   After processing the lower resolutions, the network begins upsampling, essentially enlarging the image. Simultaneously, it combines features from different scales, integrating overall body information with detailed parts.

4. **Final Prediction**

   Finally, the network generates a set of heatmaps as its prediction, highlighting different features on the map. Here, the network attempts to predict the locations of human joints in the image, such as elbows and knees.

The last part is specific to the application domain, so let's not delve into that now.

Let's redraw the model architecture:

![hourglass_2](./img/hourglass_2.jpg)

The area outlined in green represents the backbone, performing N stages of downsampling. The paper mentions that this downsampling involves a series of convolutional layers combined with max-pooling operations. Then, the upsampling process starts, incrementally increasing the feature map resolution and summing them up. Channel alignment is achieved through 1×1 convolutions.

By now, it should be quite clear:

In terms of feature fusion, Hourglass is essentially FPN, and FPN is Hourglass!

Of course, their application scenarios differ. Hourglass primarily uses the highest resolution feature map (P1) and stacks multiple layers to extract key points, whereas FPN doesn't specifically emphasize stacking many layers (although it's practical) and focuses on using different resolution feature maps (P3, P4, P5) for multi-scale object detection.

So why is there such a discrepancy in citations between these two papers?

We can only speculate: object detection is more popular, which could be one reason. Another potential reason is the narrative structure of the papers. The Hourglass paper mainly emphasizes "stacking many layers" and "intermediate supervision."

## The Origin of Hourglass?

A closer look at the Hourglass paper reveals an earlier reference with a similar structure:

**[Bottom-Up and Top-Down Reasoning with Hierarchical Rectified Gaussians (July 2015)](https://arxiv.org/abs/1507.05699)**

This earlier paper doesn't refer to the architecture as "Hourglass," but rather as a "Bottom-Up" and "Top-Down" structure. Here's a quote from the paper:

> The main purpose of this paper is to explore a "bidirectional" structure that considers top-down feedback: neurons are influenced by neurons both above and below them. This paper proposes treating neurons as rectified latent variables within a quadratic energy function, akin to a hierarchical rectified Gaussian model (RGs). The authors show that RGs can be optimized through quadratic programming (QP), which can be solved by a recurrent neural network with rectified linear units. This allows RGs to be trained using GPU-optimized gradient descent.

Wait, what does that mean?

To simplify:

This paper explores a new method to handle neurons more precisely, akin to fine-tuning them for specific tasks. This method is applied in a mathematical model that better handles data like images. The researchers show that this method can be optimized by solving a special mathematical problem, which can be solved by a specific recurrent neural network. This structure allows more efficient use of hardware resources during computation.

If it's still unclear, don't worry; it's not the main focus here. We're just trying to find the earliest reference to this architecture. Below is the model structure they used:

![hourglass_3](./img/hourglass_3.jpg)

Seeing this familiar structure, one might agree that it could have inspired the Hourglass design. But is this the earliest design?

Not necessarily.

Because six months before this paper:

- **[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)**

was proposed. The multi-scale connection structure in U-Net is quite similar, with the difference being that U-Net uses concatenation. Unfortunately, the Hourglass paper doesn't cite U-Net, so we can't see the authors' summary and evaluation of this previous work.

## Conclusion

Although Hourglass and FPN are architectures applied in different fields, their core concept of feature fusion is essentially the same, just different applications of the same architecture.

On the same foundational architecture, different application processes have evolved to solve different problems. FPN handles multi-scale information through a feature pyramid, mainly used for object detection and segmentation. In contrast, Hourglass networks extract features at various detail levels, particularly suitable for dense prediction tasks like pose estimation.

If you've ever been confused about the fundamental nature of these two architectures, hopefully, this article has provided some clarity.
