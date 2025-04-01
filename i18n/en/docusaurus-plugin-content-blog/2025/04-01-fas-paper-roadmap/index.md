---
slug: fas-paper-roadmap
title: Face Anti-Spoofing Technology Map
authors: Z. Yuan
image: /en/img/2025/0401.jpg
tags: [face-anti-spoofing, liveness-detection]
description: A guide to 40 papers from traditional to future advancements.
---

What is Face Anti-Spoofing? Why is it important? How do I get started?

This article is a comprehensive roadmap I’ve put together after reading a substantial amount of literature, designed for those who are learning, researching, or developing FAS systems.

I have selected the 40 most representative papers, divided into eight major themes based on time and technological advancements. Each paper includes reasons to read, key contributions, and the appropriate positioning. From traditional LBP, rPPG, and CNN to Transformer, CLIP, and Vision-Language Models, you will get the full scope.

Later, I will share the details of each paper in the "Paper Notes" section. Let’s first get a grasp of the overall context.

<!-- truncate -->

## Chapter 1: The Dawn of Low-Resolution Light

> **From traditional feature engineering to the first glimmer of deep learning**

Early research on Face Anti-Spoofing primarily relied on traditional image processing techniques. Researchers used handcrafted features such as texture, contrast, and frequency to describe the authenticity of faces, performing binary classification with classic classifiers.

1. [**[10.09] Face Liveness Detection from a Single Image with Sparse Low Rank Bilinear Discriminative Model**](https://parnec.nuaa.edu.cn/_upload/article/files/4d/43/8a227f2c46bda4c20da97715f010/db1eef47-b25f-4af9-88d4-a8afeccda889.pdf)
   Using the Lambertian model and sparse low-rank representation to construct feature space, effectively separating real faces from photos, providing theoretical and practical basis for early single-image liveness detection.

2. [**[12.09] On the Effectiveness of Local Binary Patterns in Face Anti-Spoofing**](https://ieeexplore.ieee.org/document/6313548)
   Utilizing LBP and its variants, this paper recognizes flat photos and screen replay attacks and establishes the REPLAY-ATTACK dataset, one of the earliest publicly available datasets and classic baselines.

3. [**[14.05] Spoofing Face Recognition with 3D Masks**](https://ieeexplore.ieee.org/document/6810829)
   A systematic analysis of the attack effects of 3D masks on different face recognition systems (2D/2.5D/3D), pointing out that the traditional assumption of flat fake faces is no longer valid with 3D printing technologies.

4. [**[19.09] Biometric Face Presentation Attack Detection with Multi-Channel Convolutional Neural Network**](https://arxiv.org/abs/1909.08848)
   Proposing a multi-channel CNN architecture that combines RGB, depth, infrared, and thermal signals for recognition, and releasing the WMCA dataset to enhance detection of advanced fake faces (e.g., silicone masks).

5. [**[22.10] Deep Learning for Face Anti-Spoofing: A Survey**](https://ieeexplore.ieee.org/abstract/document/9925105)
   The first systematic survey in the FAS field focusing on deep learning, covering pixel-wise supervision, multi-modal sensors, and domain generalization trends, establishing a comprehensive knowledge base.

---

Although these methods are simple, they laid the foundation for recognizing flat fake faces (e.g., photos and screen replays) and set the conceptual framework for the later introduction of deep learning techniques.

## Chapter 2: The Real-World Stage

> **A milestone for FAS technology moving from the lab to real-world scenarios**

Datasets and benchmarks determine whether a field can grow steadily.

FAS technology expanded from a single scene to multiple devices, lighting conditions, and attack methods, driven by these representative public datasets.

6. [**[17.06] OULU-NPU: A Mobile Face Presentation Attack Database with Real-World Variations**](https://ieeexplore.ieee.org/document/7961798)
   A mobile-specific FAS dataset designed for real-world factors such as device, environmental lighting, and attack methods, with four testing protocols, becoming a milestone in "generalization ability" evaluation.

7. [**[20.03] CASIA-SURF CeFA: A Benchmark for Multi-modal Cross-ethnicity Face Anti-Spoofing**](https://arxiv.org/abs/2003.05136)
   The world’s first large-scale multi-modal FAS dataset with "ethnicity annotations," covering RGB, Depth, IR, and multiple attack types, specifically used to study ethnic bias and modality fusion strategies.

8. [**[20.07] CelebASpoof: Large-scale Face Anti-Spoofing Dataset with Rich Annotations**](https://arxiv.org/abs/2007.12342)
   The largest FAS dataset currently, with over 620,000 images and 10 types of spoof annotations, along with 40 attributes from the original CelebA, enabling multi-task and spoof trace learning.

9. [**[22.01] A Personalized Benchmark for Face Anti-Spoofing**](https://openaccess.thecvf.com/content/WACV2022W/MAP-A/html/Belli_A_Personalized_Benchmark_for_Face_Anti-Spoofing_WACVW_2022_paper.html)
   Advocating for including liveness images from user registration in the recognition process, proposing two new test configurations, CelebA-Spoof-Enroll and SiW-Enroll, exploring the possibility of personalized FAS systems.

10. [**[24.02] SHIELD: An Evaluation Benchmark for Face Spoofing and Forgery Detection with Multimodal Large Language Models**](https://arxiv.org/abs/2402.04178)
    Combining LLM and multi-modal inputs, proposing a QA task format to evaluate the reasoning ability of MLLMs in spoof/forgery detection, opening a new field of "understanding attacks with language modeling."

## Chapter 3: The Cross-Domain Battleground

> **From single-domain learning to core technologies for multi-scene deployment**

One of the most challenging problems in Face Anti-Spoofing is generalization—how to make models not only effective on training data but also capable of handling new devices, environments, and attacks.

11. [**[20.04] Single-Side Domain Generalization for Face Anti-Spoofing**](https://arxiv.org/abs/2004.14043)
    Proposing a one-sided adversarial learning strategy, aligning only real faces across domains, allowing fake face features to naturally scatter across domains, and preventing over-compression of erroneous information. This is an enlightening direction for DG design.

12. [**[21.05] Generalizable Representation Learning for Mixture Domain Face Anti-Spoofing**](https://arxiv.org/abs/2105.02453)
    Not assuming known domain labels, but using instance normalization and MMD for unsupervised clustering and alignment, achieving a generalization training process that does not rely on manual grouping.

13. [**[23.03] Rethinking Domain Generalization for Face Anti-Spoofing: Separability and Alignment**](https://arxiv.org/abs/2303.13662)
    Proposing the SA-FAS framework, emphasizing maintaining feature separability across different domains while ensuring that the live-to-spoof transition path is consistent across domains, a deep application of IRM theory in FAS.

14. [**[24.02] Suppress and Rebalance: Towards Generalized Multi-Modal Face Anti-Spoofing**](https://arxiv.org/abs/2402.19298)
    A deep analysis of the multi-modal DG problem, using U-Adapter to suppress unstable modal interference, paired with ReGrad to dynamically adjust the convergence speed of each modality, providing a complete solution for modality imbalance and reliability issues.

15. [**[24.04] VL-FAS: Domain Generalization via Vision-Language Model for Face Anti-Spoofing**](https://ieeexplore.ieee.org/document/10448156)
    Introducing Vision-Language mechanisms for the first time, guiding attention to face regions via semantic guidance, combined with image-text contrastive learning (SLVT) for semantic layer generalization, significantly improving ViT's cross-domain stability.

---

These five papers form the core technical axis under the current Domain Generalization (DG) theme, from one-sided adversarial, label-free clustering, separability analysis, to supervisory methods that integrate language, presenting a complete strategy to address cross-domain challenges.

## Chapter 4: The Rise of a New World

> **From CNN to ViT, the architectural innovation path of FAS models**

The rise of Vision Transformers (ViT) has ushered in an era of global modeling for image tasks, shifting away from local convolutions. Face Anti-Spoofing (FAS) is no exception.

16. [**[23.02] Rethinking Vision Transformer and Masked Autoencoder in Multimodal Face Anti-Spoofing**](https://arxiv.org/abs/2302.05744)
    A comprehensive review of the core issues of ViT in multimodal FAS, including input design, pre-training strategies, and fine-tuning processes. The paper proposes the AMA adapter and M2A2E pre-training architecture to construct cross-modal, label-free self-supervised workflows.

17. [**[23.04] Ma-ViT: Modality-Agnostic Vision Transformers for Face Anti-Spoofing**](https://arxiv.org/abs/2304.07549)
    Using a single-branch early fusion architecture, this paper implements modality-agnostic recognition ability through Modal-Disentangle Attention and Cross-Modal Attention, balancing memory efficiency and flexible deployment, marking an important step in ViT's practicality.

18. [**[23.05] FM-ViT: Flexible Modal Vision Transformers for Face Anti-Spoofing**](https://arxiv.org/abs/2305.03277)
    To solve the issues of modality loss and high-fidelity attacks, the paper introduces a cross-modal attention design (MMA + MFA), which strengthens the focus on spoof patches while preserving the characteristics of each modality, serving as a model for deployment flexibility.

19. [**[23.09] Sadapter: Generalizing Vision Transformer for Face Anti-Spoofing with Statistical Tokens**](https://arxiv.org/abs/2309.04038)
    Using an Efficient Parameter Transfer Learning architecture, this approach inserts statistical adapters into ViT while fixing the main network parameters. Token Style Regularization helps suppress style differences, providing a lightweight solution for cross-domain FAS.

20. [**[23.10] LDCFormer: Incorporating Learnable Descriptive Convolution to Vision Transformer for Face Anti-Spoofing**](https://ieeexplore.ieee.org/document/10222330)
    Combining learnable descriptive convolution (LDC) with ViT to enhance local detail representation, the paper introduces a decoupled optimization version (LDCformerD), achieving state-of-the-art performance across multiple benchmarks.

---

These five papers demonstrate how the Transformer architecture handles critical challenges in multimodal input, modality loss, cross-domain style, and local patch representations, representing a comprehensive shift in the logic of FAS model design.

## Chapter 5: The Battle of Styles

> **When spoofing comes from different worlds, how can we build style-invariant models?**

The generalization of FAS models is not only challenged by domain shifts but also by the interference caused by asymmetric information between different styles.

This chapter focuses on style decoupling, adversarial learning, test-time adaptation, and instance-aware designs. These approaches attempt to enable models to maintain stable recognition performance even under unknown styles and sample distributions.

21. [**[21.07] Unified Unsupervised and Semi-Supervised Domain Adaptation Network for Cross-Scenario Face Anti-Spoofing**](https://www.sciencedirect.com/science/article/abs/pii/S0031320321000753)
    Proposing the USDAN framework, which supports both unsupervised and semi-supervised settings, and learns generalized representations compatible with different task configurations through marginal and conditional alignment modules, along with adversarial training.

22. [**[22.03] Domain Generalization via Shuffled Style Assembly for Face Anti-Spoofing**](https://arxiv.org/abs/2203.05340)
    Adopting content and style separation strategies, this paper reshuffles the style space to simulate style shifts, emphasizing live-related styles through contrastive learning. It represents a significant breakthrough in style-aware domain generalization (DG) design.

23. [**[23.03] Adversarial Learning Domain-Invariant Conditional Features for Robust Face Anti-Spoofing**](https://link.springer.com/article/10.1007/s11263-023-01778-x)
    Not only aligning marginal distributions, but also introducing adversarial structures for conditional alignment, learning distinguishable cross-domain shared representations at the class level, effectively solving misalignment issues.

24. [**[23.03] Style Selective Normalization with Meta Learning for Test-Time Adaptive Face Anti-Spoofing**](https://www.sciencedirect.com/science/article/abs/pii/S0957417422021248)
    Utilizing statistical information to estimate the style of input images, this method dynamically selects normalization parameters for test-time adaptation, and combines meta-learning to pre-simulate the transfer process for unknown domains.

25. [**[23.04] Instance-Aware Domain Generalization for Face Anti-Spoofing**](https://arxiv.org/abs/2304.05640)
    Discarding coarse domain labels, this paper adopts an instance-level style alignment strategy, refining style-invariant recognition features through asymmetric whitening, style enhancement, and dynamic kernel designs.

---

These five papers challenge the "style generalization" theme from different angles, particularly with attempts at instance-based and test-time adaptation, gradually approaching the demands of real-world applications.

## Chapter 6: The Summoning of Multimodality

> **When images are no longer the only modality, sound and physiological signals come into play**

When traditional RGB models face bottlenecks in high-fidelity attacks and cross-domain challenges, the FAS community began exploring non-visual signals, such as **rPPG, physiological signals, and acoustic echoes**, to establish recognition bases that are harder to forge, starting from "human-centered signals."

This chapter features five representative papers spanning physiological signals, 3D geometry, and acoustic perception, showcasing the potential and future of multimodal FAS technology.

26. [**[18.09] Remote Photoplethysmography Correspondence Feature for 3D Mask Face Presentation Attack Detection**](https://dl.acm.org/doi/10.1007/978-3-030-01270-0_34)
    Introducing CFrPPG (Correspondence rPPG) features to enhance liveness signal acquisition, ensuring accurate heart rate tracking even under low light or camera shake, showing strong performance against 3D mask attacks.

27. [**[19.05] Multi-Modal Face Authentication Using Deep Visual and Acoustic Features**](https://ieeexplore.ieee.org/document/8761776)
    Using the built-in speakers and microphones of smartphones, this method emits ultrasound and analyzes facial echoes, combined with CNN-extracted image features, creating a dual-modal authentication system that requires no additional hardware.

28. [**[21.04] Contrastive Context-Aware Learning for 3D High-Fidelity Mask Face Presentation Attack Detection**](https://arxiv.org/abs/2104.06148)
    To address the challenge of high-fidelity 3D masks, the HiFiMask dataset is introduced, along with a Contrastive Context-Aware Learning method, using context information (person, material, lighting) to enhance attack detection capability.

29. [**[22.08] Beyond the Pixel World: A Novel Acoustic-Based Face Anti-Spoofing System for Smartphones**](https://ieeexplore.ieee.org/document/9868051)
    Creating the Echo-Spoof acoustic FAS dataset and designing the Echo-FAS framework, which uses sound waves to reconstruct 3D geometry and material information, entirely independent of cameras, showcasing a low-cost and high-resilience mobile device application.

30. [**[24.03] AFace: Range-Flexible Anti-Spoofing Face Authentication via Smartphone Acoustic Sensing**](https://dl.acm.org/doi/10.1145/3643510)
    Extending the Echo-FAS concept, incorporating an iso-depth model and distance-adaptive algorithm to combat 3D printed masks, and adjusting based on user distance, this is a crucial design in the practical implementation of acoustic-based liveness verification.

---

These five papers mark the beginning of the significant role non-image modalities play in FAS, and if you wish to bypass the limitations of traditional cameras, this is a promising direction worth exploring.

## Chapter 7: Decoding the Trace of Deception

> **Deeply modeling the structure and semantics of spoofing to enhance model discriminability**

As FAS models face dual challenges of interpretability and generalization, researchers have begun to focus on the concept of "spoof trace": the subtle patterns left by fake faces in images, such as color biases, edge contours, or frequency anomalies.

The five papers in this chapter all approach this from the perspective of **representation disentanglement**, attempting to separate spoof features from facial content, then reconstruct, analyze, or even synthesize spoof samples, allowing models to truly "see through the disguise."

31. [**[20.07] On Disentangling Spoof Trace for Generic Face Anti-Spoofing**](https://arxiv.org/abs/2007.09273)
    Proposes a multi-scale spoof trace separation model, treating spoof signals as multi-layered patterns. Through adversarial learning, it reconstructs real faces and spoof masks, applicable for synthesizing new attack samples. It is a representative work in spoof-aware representation learning.

32. [**[20.08] Face Anti-Spoofing via Disentangled Representation Learning**](https://arxiv.org/abs/2008.08250)
    Decomposes facial features into two subspaces: liveness and identity. Through a CNN structure, it separates low- and high-level signals to build a more transferable liveness classifier, improving stability across different attack types.

33. [**[22.03] Spoof Trace Disentanglement for Generic Face Anti-Spoofing**](https://ieeexplore.ieee.org/document/9779478)
    Models spoof traces as additive and repairable patterns, proposing a two-stage disentanglement framework that incorporates frequency domain information to strengthen low-level spoof detection, also useful for spoof data augmentation to enhance long-tail attack generalization.

34. [**[22.07] Learning to Augment Face Presentation Attack Dataset via Disentangled Feature Learning from Limited Spoof Data**](https://ieeexplore.ieee.org/document/9859657)
    Proposes a disentangled remix strategy for limited spoof samples, generating in the separated liveness and identity feature spaces, and using contrastive learning to maintain discriminability, significantly improving recognition performance in small-sample scenarios.

35. [**[22.12] Learning Polysemantic Spoof Trace: A Multi-Modal Disentanglement Network for Face Anti-Spoofing**](https://arxiv.org/abs/2212.03943)
    Extends the spoof trace disentanglement framework to multimodal settings, designing an RGB/Depth dual-network to capture complementary spoof clues and integrating cross-modality fusion to combine their semantics, offering a forward-looking solution for universal FAS models.

---

This chapter marks a key turning point: from recognizing liveness → analyzing disguises → simulating attacks, Face Anti-Spoofing research is gradually moving toward the next stage of "generative, interpretable, and controllable" models. These methods not only improve model accuracy but may also inspire the future evolution of offense and defense strategies.

## Chapter 8: The Chaotic Landscape of the Future

> **From CLIP to human perception, the next frontier of FAS**

As single-modal and single-attack-type solutions fail to meet real-world needs, FAS is stepping into higher-level challenges: **physical + digital dual attacks, semantic-driven recognition, and zero-shot generalization in diverse environments**.

These five representative works are the three major development axes for the future of FAS: **fusion recognition, language modeling, and human-centered perception**.

36. [**[20.07] Face Anti-Spoofing with Human Material Perception**](https://arxiv.org/abs/2007.02157)
    Integrates material perception into FAS model design, with the BCN architecture simulating human perception at macro and micro levels to judge material differences (skin, paper, silicone), enhancing the model's semantic interpretability and cross-material recognition ability.

37. [**[23.09] FLIP: Cross-domain Face Anti-Spoofing with Language Guidance**](https://arxiv.org/abs/2309.16649)
    Applies the CLIP model to the FAS task, guiding visual representation spaces through natural language descriptions to improve cross-domain generalization. The paper proposes semantic alignment and multimodal contrastive learning strategies, achieving true zero-shot FAS under language guidance.

38. [**[24.04] Joint Physical-Digital Facial Attack Detection via Simulating Spoofing Clues**](https://arxiv.org/abs/2404.08450)
    Proposes SPSC and SDSC data augmentation strategies to simulate both physical and digital attack clues, enabling a single model to learn to recognize both types of attacks. This won the CVPR 2024 competition, setting a new paradigm for fusion models.

39. [**[24.04] Unified Physical-Digital Attack Detection Challenge**](https://arxiv.org/abs/2404.06211)
    Launched the first unified attack detection challenge, releasing the 28,000-entry UniAttackData complex attack dataset and analyzing model architectures, catalyzing the research community toward Unified Attack Detection.

40. [**[24.08] La-SoftMoE CLIP for Unified Physical-Digital Face Attack Detection**](https://arxiv.org/abs/2408.12793)
    Combines CLIP with the Mixture of Experts architecture, introducing a soft-adaptive mechanism to dynamically assign sub-models for complex decision boundaries, providing an efficient parameter selection solution for physical and digital attack fusion handling.

---

This chapter signifies the future trend in the FAS field: **from recognizing fake faces → inferring attack types → understanding semantics → combining multimodal language logic reasoning**. Research is evolving from "visual understanding" to "semantic cognition," and attacks are shifting from single-mode to complex hybrid models.

## Conclusion

The real world is never short of malice. As long as there is a demand for face recognition, the need for anti-spoofing will never stop.

From the initial texture analysis and light-shadow modeling to the advent of convolutional networks, and now to the introduction of ViT, CLIP, sound waves, and human perception, FAS technology continues to expand its boundaries. These papers are not only a collection of classics and trends but also a map that spans decades of technological evolution, connecting the past, present, and future.

On this map, we see:

- **From single-modal to multimodal**: Not just seeing the image but sensing depth, sound, pulse, and material.
- **From classification to disentanglement**: Not just determining real or fake, but attempting to understand the structure of each disguise.
- **From recognition to reasoning**: Not just distinguishing liveness, but starting to understand the semantics, materials, and language descriptions behind the truth.
- **From defense to generation**: Not just passive defense, but starting to simulate, reconstruct, and intervene proactively.

If you're planning to enter this field, this technical guide won't give you "a one-size-fits-all solution," but it will help you find your starting point: are you fascinated by the visualization of spoof traces? Or do you want to explore how CLIP can assist in secure recognition? Or perhaps you're interested in sound waves and material recognition?

No matter what your background is, FAS is an intersection of image recognition, biometrics, human perception, semantic reasoning, and cross-modal fusion.

This battle is far from over.
