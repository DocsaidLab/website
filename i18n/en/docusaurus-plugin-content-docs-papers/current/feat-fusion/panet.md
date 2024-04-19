---
sidebar_position: 2
---

# PANet

## Give Me a Shortcut

**[Path Aggregation Network for Instance Segmentation (2018.03)](https://arxiv.org/abs/1803.01534v4)**

---

:::info
The following content is compiled by ChatGPT-4, with manual proofreading, editing, and additional explanations.
:::

---

After the classic FPN was introduced, the question of "how to perform feature fusion more efficiently?" became a topic worthy of research.

Let's take a look at another classic architecture — PANet.

## Defining the Problem

In the PANet paper, the main comparison is made with the FPN architecture. If you're not familiar with it, you can refer to:

- **Link: [FPN: Feature Pyramid Networks](./fpn)**

Recall our earlier conclusion about FPN: "Bottom-up, top-down, then add them together."

For this design, PANet considers it insufficient. Here's a quote from the authors:

>The insightful point that neurons in high layers strongly respond to entire objects while other neurons are more likely to be activated by local texture and patterns manifests the necessity of augmenting a top-down path to propagate semantically strong features and enhance all features with reasonable classification capability in FPN.
>
> Excerpted from PANet

I believe the beauty of the original sentiment would be lost if translated directly into Chinese, so I opted to leave it as is. In simple terms, neurons in high layers can see entire objects, while neurons in lower layers focus on local texture and patterns. Therefore, a single pathway for feature fusion is insufficient; we need to add another one.

This quote encapsulates the core functionality of the entire paper.

![panet_1](./resources/fpn_2.jpg)

Let's break down this quote with the help of the image. First, "neurons in high layers" refer to Block5. These neurons have a large receptive field and can perceive large objects. On the other hand, features from lower layers like P1, P2, etc., focus on local texture and patterns.

The original FPN design allows lower-level features (e.g., P1) to reference higher-level features (e.g., P5) with a short pathway, perhaps consisting of only a few or a dozen convolutional layers. However, for higher-level features (e.g., P5) to reference lower-level features (e.g., P1), it might require hundreds of convolutional layers, depending on the chosen backbone network.

That's quite a disparity, isn't it? Shouldn't we make some adjustments?

## Solving the Problem

### PANet Model Design

![panet_2](./resources/panet_2.jpg)

As shown in the image above, this is the improvement proposed by the authors. First is image (a), which represents the FPN architecture we discussed earlier. Then we have image (b), where the authors added a path: if the original path is too long (depicted by the red line), provide a shortcut from bottom to top (depicted by the green line). Images (c) and (d) belong to the Head structure, which is not within the scope of our discussion here, so we'll skip them.

### Path Aggregation Network (PANet)

Lastly, let's delve into how the Path Aggregation Modules are implemented: for each building block, we combine the features from the lower-level feature map (Ni) and the higher-level feature map (Pi+1) to create a new feature map (Ni+1).

For each feature map (Ni), we first use a 3x3 convolution operation with stride=2 to reduce the spatial dimensions of the image by half, allowing us to process less data. It's important to note that N2 in Figure 1 is just P2 without any processing.

Next, we add each small block from the higher-level feature map (Pi+1) to the corresponding position in the lower-level feature map (Ni). Finally, we apply another 3x3 convolution operation to this combined feature map to obtain a new feature map (Ni+1) for subsequent subnetworks.

This process is repeated until we reach the feature map close to the top level (P5), at which point it stops. In these building blocks, PANet maintains the channel number of features at 256, and each convolution operation is followed by a ReLU activation function, which the authors believe allows the model to learn more useful features.

Finally, from these newly generated feature maps, the features of each proposal grid are combined together, forming the [N2, N3, N4, N5] feature grids shown in the image. Through this approach, the PANet architecture not only preserves the importance of features but also effectively enhances the flow of information, laying a solid foundation for subsequent tasks.

## Discussion

### How Does It Compare to Basic FPN?

![panet_3](./resources/panet_3.jpg)

As we delve deeper into this architecture, we can see that the authors conducted a lot of analysis and experiments during the optimization process.

Let's highlight some key points together.

First, let's focus on the "Multi-Scale Training" element, denoted as "MST" in the image.

This is the implementation of the FPN architecture mentioned earlier, and it's evident that when this element is added, the original model's performance improves by 2% in terms of AP score.

Next, the authors added "Multi-GPU BatchNorm." Starting with the original multi-scale architecture, this further improved performance by 0.4%. Then comes the crucial element emphasized in this paper: "Bottom-Up Path Augmentation" (BPA), represented in the image as "BPA." The core idea of this method is to enhance paths from the bottom, combining lower-level feature maps with higher-level ones to establish a richer hierarchical structure of information.

After integrating "BPA," the model's performance improved by another 0.7%. We feel it's a bit unfortunate that a "Dilution Experiment" was inserted between "MST" and "BPA," so we couldn't see a direct comparison between "MST" alone and "MST+BPA."

In addition to "Bottom-Up Path Augmentation," the authors introduced several other key elements to make the entire architecture more robust. While these belong to the Model Head part and are not the focus of this paper, since the diagrams are provided, let's briefly go over some information.

One of them is "Adaptive Feature Pooling," which connects feature grids with all feature levels, allowing useful information from each level to propagate directly to subsequent subnetworks. This approach enables better cooperation between features of different levels, further enhancing the model's performance. Additionally, "Fully Connected Fusion" is also an important component. Through this method, the authors aim to improve the model's prediction quality, thereby enhancing overall performance.

## Conclusion

PANet continues the design philosophy of FPN, further exploring how to better address the problem of multi-scale feature fusion while innovating further. This architecture not only effectively combines various backbone networks to build a stronger feature pyramid but also provides superior performance improvements for multi-scale tasks.

The PANet paper conveys two important messages:

First, regardless of the multi-scale problem we face, we must consider how to perform feature fusion. The design concept of PANet is full of attention to features at different scales, and through path enhancement, feature pooling, and fusion methods, it integrates these scales into a powerful feature representation.

Second, unlike what was mentioned in FPN: "Bottom-Up, Top-Down, then Add them together"; in PANet, the authors believe it should be changed to "Bottom-Up, Top-Down, Bottom-Up, then Add them together."

In the process of exploring feature fusion, there are still a series of unresolved issues, such as comparing additive fusion with concatenation fusion, improving fusion efficiency, and adjusting fusion weights. These issues will be further explored in future research.
