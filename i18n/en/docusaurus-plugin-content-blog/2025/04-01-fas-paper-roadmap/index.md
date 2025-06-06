---
slug: fas-paper-roadmap
title: Face Anti-Spoofing Technology Map
authors: Z. Yuan
image: /en/img/2025/0401.jpg
tags: [face-anti-spoofing, liveness-detection]
description: A guide to 40 papers from FAS.
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

   :::info
   **Paper Notes**：[**[10.09] SLRBD: Silent Reflective Light**](https://docsaid.org/en/papers/face-antispoofing/slrbd/)
   :::

2. [**[12.09] On the Effectiveness of Local Binary Patterns in Face Anti-Spoofing**](https://ieeexplore.ieee.org/document/6313548)
   Utilizing LBP and its variants, this paper recognizes flat photos and screen replay attacks and establishes the REPLAY-ATTACK dataset, one of the earliest publicly available datasets and classic baselines.

   :::info
   **Paper Notes**：[**[12.09] LBP: Lively Micro-textures**](https://docsaid.org/en/papers/face-antispoofing/lbp/)
   :::

3. [**[14.05] Spoofing Face Recognition with 3D Masks**](https://ieeexplore.ieee.org/document/6810829)
   A systematic analysis of the attack effects of 3D masks on different face recognition systems (2D/2.5D/3D), pointing out that the traditional assumption of flat fake faces is no longer valid with 3D printing technologies.

   :::info
   **Paper Notes**：[**[14.05] 3DMAD: The Real Mask**](https://docsaid.org/en/papers/face-antispoofing/three-d-mad/)
   :::

4. [**[19.09] Biometric Face Presentation Attack Detection with Multi-Channel Convolutional Neural Network**](https://arxiv.org/abs/1909.08848)
   Proposing a multi-channel CNN architecture that combines RGB, depth, infrared, and thermal signals for recognition, and releasing the WMCA dataset to enhance detection of advanced fake faces (e.g., silicone masks).

   :::info
   **Paper Notes**：[**[19.09] WMCA: The Invisible Face**](https://docsaid.org/en/papers/face-antispoofing/wmca/)
   :::

5. [**[22.10] Deep Learning for Face Anti-Spoofing: A Survey**](https://ieeexplore.ieee.org/abstract/document/9925105)
   The first systematic survey in the FAS field focusing on deep learning, covering pixel-wise supervision, multi-modal sensors, and domain generalization trends, establishing a comprehensive knowledge base.

   :::info
   **Paper Notes**：[**[22.10] FAS Survey: A Chronicle of Attacks and Defenses**](https://docsaid.org/en/papers/face-antispoofing/fas-survey/)
   :::

---

Although these methods are simple, they laid the foundation for recognizing flat fake faces (e.g., photos and screen replays) and set the conceptual framework for the later introduction of deep learning techniques.

## Chapter 2: The Real-World Stage

> **A milestone for FAS technology moving from the lab to real-world scenarios**

Datasets and benchmarks determine whether a field can grow steadily.

FAS technology expanded from a single scene to multiple devices, lighting conditions, and attack methods, driven by these representative public datasets.

6. [**[17.06] OULU-NPU: A Mobile Face Presentation Attack Database with Real-World Variations**](https://ieeexplore.ieee.org/document/7961798)
   A mobile-specific FAS dataset designed for real-world factors such as device, environmental lighting, and attack methods, with four testing protocols, becoming a milestone in "generalization ability" evaluation.

   :::info
   **Paper Notes**: [**[17.06] OULU-NPU: Four Challenges**](https://docsaid.org/en/papers/face-antispoofing/oulu-npu/)
   :::

7. [**[20.03] CASIA-SURF CeFA: A Benchmark for Multi-modal Cross-ethnicity Face Anti-Spoofing**](https://arxiv.org/abs/2003.05136)
   The world’s first large-scale multi-modal FAS dataset with "ethnicity annotations," covering RGB, Depth, IR, and multiple attack types, specifically used to study ethnic bias and modality fusion strategies.

   :::info
   **Paper Notes**: [**[20.03] CeFA: Discrimination in Models**](https://docsaid.org/en/papers/face-antispoofing/cefa/)
   :::

8. [**[20.07] CelebASpoof: Large-scale Face Anti-Spoofing Dataset with Rich Annotations**](https://arxiv.org/abs/2007.12342)
   The largest FAS dataset currently, with over 620,000 images and 10 types of spoof annotations, along with 40 attributes from the original CelebA, enabling multi-task and spoof trace learning.

   :::info
   **Paper Notes**: [**[20.07] CelebA-Spoof: Large-Scale Anti-Spoofing Trials**](https://docsaid.org/en/papers/face-antispoofing/celeba-spoof/)
   :::

9. [**[22.01] A Personalized Benchmark for Face Anti-Spoofing**](https://openaccess.thecvf.com/content/WACV2022W/MAP-A/html/Belli_A_Personalized_Benchmark_for_Face_Anti-Spoofing_WACVW_2022_paper.html)
   Advocating for including liveness images from user registration in the recognition process, proposing two new test configurations, CelebA-Spoof-Enroll and SiW-Enroll, exploring the possibility of personalized FAS systems.

   :::info
   **Paper Notes**: [**[22.01] Personalized-FAS: Personalized Attempt**](https://docsaid.org/en/papers/face-antispoofing/personalized-fas/)
   :::

10. [**[24.02] SHIELD: An Evaluation Benchmark for Face Spoofing and Forgery Detection with Multimodal Large Language Models**](https://arxiv.org/abs/2402.04178)
    Combining LLM and multi-modal inputs, proposing a QA task format to evaluate the reasoning ability of MLLMs in spoof/forgery detection, opening a new field of "understanding attacks with language modeling."

    :::info
    **Paper Notes**: [**[24.02] SHIELD: Tell me, why?**](https://docsaid.org/en/papers/face-antispoofing/shield/)
    :::

## Chapter 3: The Cross-Domain Battleground

> **From single-domain learning to core technologies for multi-scene deployment**

One of the most challenging problems in Face Anti-Spoofing is generalization—how to make models not only effective on training data but also capable of handling new devices, environments, and attacks.

11. [**[20.04] Single-Side Domain Generalization for Face Anti-Spoofing**](https://arxiv.org/abs/2004.14043)
    Proposing a one-sided adversarial learning strategy, aligning only real faces across domains, allowing fake face features to naturally scatter across domains, and preventing over-compression of erroneous information. This is an enlightening direction for DG design.

    :::info
    **Paper Notes**: [**[20.04] SSDG: Stable Realness**](https://docsaid.org/en/papers/face-antispoofing/ssdg/)
    :::

12. [**[21.05] Generalizable Representation Learning for Mixture Domain Face Anti-Spoofing**](https://arxiv.org/abs/2105.02453)
    Not assuming known domain labels, but using instance normalization and MMD for unsupervised clustering and alignment, achieving a generalization training process that does not rely on manual grouping.

    :::info
    **Paper Notes**: [**[21.05] D²AM: Thousand-Domain Soul Forging**](https://docsaid.org/en/papers/face-antispoofing/d2am/)
    :::

13. [**[23.03] Rethinking Domain Generalization for Face Anti-Spoofing: Separability and Alignment**](https://arxiv.org/abs/2303.13662)
    Proposing the SA-FAS framework, emphasizing maintaining feature separability across different domains while ensuring that the live-to-spoof transition path is consistent across domains, a deep application of IRM theory in FAS.

    :::info
    **Paper Notes**: [**[23.03] SA-FAS: The Law of the Hyperplane**](https://docsaid.org/en/papers/face-antispoofing/sa-fas/)
    :::

14. [**[24.02] Suppress and Rebalance: Towards Generalized Multi-Modal Face Anti-Spoofing**](https://arxiv.org/abs/2402.19298)
    A deep analysis of the multi-modal DG problem, using U-Adapter to suppress unstable modal interference, paired with ReGrad to dynamically adjust the convergence speed of each modality, providing a complete solution for modality imbalance and reliability issues.

    :::info
    **Paper Notes**: [**[24.02] MMDG: Trust Management**](https://docsaid.org/en/papers/face-antispoofing/mmdg/)
    :::

15. [**[24.03] CFPL-FAS: Class Free Prompt Learning for Generalizable Face Anti-spoofing**](https://arxiv.org/abs/2403.14333)
    Focuses on a prompt learning approach that emphasizes class-free prompt design, eliminating the need for manually defined categories. This represents a new direction in leveraging language prompts to enhance the generalization ability of FAS models.

    :::info
    **Paper Notes**: [**[24.03] CFPL-FAS: Class-Free Prompt Learning**](https://docsaid.org/en/papers/face-antispoofing/cfpl-fas/)
    :::

---

These five papers form the core technical axis under the current Domain Generalization (DG) theme, from one-sided adversarial, label-free clustering, separability analysis, to supervisory methods that integrate language, presenting a complete strategy to address cross-domain challenges.

## Chapter 4: The Rise of a New World

> **From CNN to ViT, the architectural innovation path of FAS models**

The rise of Vision Transformers (ViT) has ushered in an era of global modeling for image tasks, shifting away from local convolutions. Face Anti-Spoofing (FAS) is no exception.

16. [**[23.01] Domain Invariant Vision Transformer Learning for Face Anti-Spoofing**](https://openaccess.thecvf.com/content/WACV2023/papers/Liao_Domain_Invariant_Vision_Transformer_Learning_for_Face_Anti-Spoofing_WACV_2023_paper.pdf)
    Proposed the DiVT architecture, which enhances cross-domain generalization through two core loss functions. It aggregates genuine face features to form more consistent domain-invariant representations. Experiments show that DiVT achieves state-of-the-art results on various DG-FAS tasks, offering a streamlined yet effective approach to capturing key information for cross-domain recognition.

    :::info
    **Paper Notes**: [**[23.01] DiVT: All-Star Championship**](https://docsaid.org/en/papers/face-antispoofing/divt/)
    :::

17. [**[23.02] Rethinking Vision Transformer and Masked Autoencoder in Multimodal Face Anti-Spoofing**](https://arxiv.org/abs/2302.05744)
    A comprehensive review of the core issues of ViT in multimodal FAS, including input design, pre-training strategies, and fine-tuning processes. The paper proposes the AMA adapter and M2A2E pre-training architecture to construct cross-modal, label-free self-supervised workflows.

    :::info
    **Paper Notes**: [**[23.02] M²A²E: Drawing Parallels**](https://docsaid.org/en/papers/face-antispoofing/m2a2e/)
    :::

18. [**[23.04] MA-ViT: Modality-Agnostic Vision Transformers for Face Anti-Spoofing**](https://arxiv.org/abs/2304.07549)
    Using a single-branch early fusion architecture, this paper implements modality-agnostic recognition ability through Modal-Disentangle Attention and Cross-Modal Attention, balancing memory efficiency and flexible deployment, marking an important step in ViT's practicality.

    :::info
    **Paper Notes**: [**[23.04] MA-ViT: All appearances Are Illusions**](https://docsaid.org/en/papers/face-antispoofing/ma-vit/)
    :::

19. [**[23.09] S-Adapter: Generalizing Vision Transformer for Face Anti-Spoofing with Statistical Tokens**](https://arxiv.org/abs/2309.04038)
    Using an Efficient Parameter Transfer Learning architecture, this approach inserts statistical adapters into ViT while fixing the main network parameters. Token Style Regularization helps suppress style differences, providing a lightweight solution for cross-domain FAS.

    :::info
    **Paper Notes**: [**[23.09] S-Adapter: Real Notebook**](https://docsaid.org/en/papers/face-antispoofing/s-adapter/)
    :::

20. [**[24.10] FM-CLIP: Flexible Modal CLIP for Face Anti-Spoofing**](https://dl.acm.org/doi/pdf/10.1145/3664647.3680856)
    By utilizing Cross-Modal Spoofing Enhancer (CMS-Enhancer) and text-guided (LGPA) dynamic alignment of spoofing cues, it maintains high detection accuracy across multi-modal training and single or multi-modal testing, demonstrating excellent generalization ability across multiple datasets.

    :::info
    **Paper Notes**: [**[24.10] FM-CLIP: Guidance from Language**](https://docsaid.org/en/papers/face-antispoofing/fm-clip/)
    :::

---

These five papers demonstrate how the Transformer architecture handles critical challenges in multimodal input, modality loss, cross-domain style, and local patch representations, representing a comprehensive shift in the logic of FAS model design.

## Chapter 5: The Battle of Styles

> **When spoofing comes from different worlds, how to build a style-agnostic model?**

The generalization of FAS models is challenged not only by domain shift but also by the interference caused by the asymmetry of information between different styles.

This chapter focuses on style disentanglement, adversarial learning, test-time adaptation, and instance-aware design. These methods aim to help the model maintain stable recognition performance under unknown styles and sample distributions.

21. [**[22.03] Domain Generalization via Shuffled Style Assembly for Face Anti-Spoofing**](https://arxiv.org/abs/2203.05340)
    Employs a content-style separation strategy, reorganizing the style space to simulate style shift. Combined with contrastive learning that emphasizes style related to liveness, this is a significant breakthrough in style-aware domain generalization (DG) design.

    :::info
    **Paper Notes**: [**[22.03] SSAN: The Shadow of Style**](https://docsaid.org/en/papers/face-antispoofing/ssan/)
    :::

22. [**[22.12] Cyclically Disentangled Feature Translation for Face Anti-spoofing**](https://arxiv.org/abs/2212.03651)
    Proposes CDFTN, which separates liveness and style components through adversarial learning, generating pseudo-labeled samples that combine real labels and target domain appearances. This significantly improves the accuracy and robustness of cross-domain spoof detection.

    :::info
    **Paper Notes**: [**[22.12] CDFTN: The Entanglement of Style**](https://docsaid.org/en/papers/face-antispoofing/cdftn/)
    :::

23. [**[23.04] Instance-Aware Domain Generalization for Face Anti-Spoofing**](https://arxiv.org/abs/2304.05640)
    Abandons coarse domain labels in favor of instance-level style alignment strategies. Through asymmetric whitening, style enhancement, and dynamic kernel design, this refines recognition features that are insensitive to style.

    :::info
    **Paper Notes**: [**[23.04] IADG: A Monologue of Styles**](https://docsaid.org/en/papers/face-antispoofing/iadg/)
    :::

24. [**[23.10] Towards Unsupervised Domain Generalization for Face Anti-Spoofing**](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Towards_Unsupervised_Domain_Generalization_for_Face_Anti-Spoofing_ICCV_2023_paper.html)
    Incorporates unlabeled data into the learning process, using segmentation and cross-domain similarity searching mechanisms to extract generalized representations that adapt to multiple unlabeled scenarios, achieving true unsupervised domain generalization (DG) for FAS.

    :::info
    **Paper Notes**: [**[23.10] UDG-FAS: Fragments of Style**](https://docsaid.org/en/papers/face-antispoofing/udg-fas/)
    :::

25. [**[23.11] Test-Time Adaptation for Robust Face Anti-Spoofing**](https://papers.bmvc2023.org/0379.pdf)
    Dynamically adjusts the model at inference time for new scenes, combining activation-based pseudo-labeling and contrastive learning to prevent forgetting, allowing pre-trained FAS models to self-optimize during testing, improving sensitivity to unknown attacks.

    :::info
    **Paper Notes**: [**[23.11] 3A-TTA: Surviving the Wilderness**](https://docsaid.org/en/papers/face-antispoofing/three-a-tta/)
    :::

---

These five papers challenge the theme of "style generalization" from different angles, especially with their attempts in instance-based and test-time adaptation, gradually approaching the demands of practical application scenarios.

## Chapter 6: The Summoning of Multimodality

> **When images are no longer the only modality, sound and physiological signals come into play**

When traditional RGB models face bottlenecks in high-fidelity attacks and cross-domain challenges, the FAS community began exploring non-visual signals, such as **rPPG, physiological signals, and acoustic echoes**, to establish recognition bases that are harder to forge, starting from "human-centered signals."

This chapter features five representative papers spanning physiological signals, 3D geometry, and acoustic perception, showcasing the potential and future of multimodal FAS technology.

26. [**[16.12] Generalized face anti-spoofing by detecting pulse from face videos**](https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICPR-2016/media/files/1223.pdf)
    In the early FAS (Face Anti-Spoofing) scenario, this paper demonstrated how facial heartbeat signals alone, without depth or infrared sensors, can be used to detect fake faces, highlighting the potential of rPPG.

    :::info
    **Paper Notes**: [**[16.12] rPPG: The Flicker of Life**](https://docsaid.org/en/papers/face-antispoofing/rppg/)
    :::

27. [**[18.09] Remote Photoplethysmography Correspondence Feature for 3D Mask Face Presentation Attack Detection**](https://dl.acm.org/doi/10.1007/978-3-030-01270-0_34)
    Introducing CFrPPG (Correspondence rPPG) features to enhance liveness signal acquisition, ensuring accurate heart rate tracking even under low light or camera shake, showing strong performance against 3D mask attacks.

    :::info
    **Paper Notes**: [**[18.09] CFrPPG: The Echo of a Heartbeat**](https://docsaid.org/en/papers/face-antispoofing/cfrppg)
    :::

28. [**[19.05] Multi-Modal Face Authentication Using Deep Visual and Acoustic Features**](https://ieeexplore.ieee.org/document/8761776)
    Using the built-in speakers and microphones of smartphones, this method emits ultrasound and analyzes facial echoes, combined with CNN-extracted image features, creating a dual-modal authentication system that requires no additional hardware.

    :::info
    **Paper Notes**: [**[19.05] VA-FAS: Faces in Sound Waves**](https://docsaid.org/en/papers/face-antispoofing/vafas)
    :::

29. [**[22.08] Beyond the Pixel World: A Novel Acoustic-Based Face Anti-Spoofing System for Smartphones**](https://ieeexplore.ieee.org/document/9868051)
    Creating the Echo-Spoof acoustic FAS dataset and designing the Echo-FAS framework, which uses sound waves to reconstruct 3D geometry and material information, entirely independent of cameras, showcasing a low-cost and high-resilience mobile device application.

    :::info
    **Paper Notes**: [**[22.08] Echo-FAS: The Echo of Spoofing**](https://docsaid.org/en/papers/face-antispoofing/echo-fas)
    :::

30. [**[24.03] AFace: Range-Flexible Anti-Spoofing Face Authentication via Smartphone Acoustic Sensing**](https://dl.acm.org/doi/10.1145/3643510)
    Extending the Echo-FAS concept, incorporating an iso-depth model and distance-adaptive algorithm to combat 3D printed masks, and adjusting based on user distance, this is a crucial design in the practical implementation of acoustic-based liveness verification.

    :::info
    **Paper Notes**: [**[24.03] AFace: The Boundary of Waves**](https://docsaid.org/en/papers/face-antispoofing/aface)
    :::

---

These five papers mark the beginning of the significant role non-image modalities play in FAS, and if you wish to bypass the limitations of traditional cameras, this is a promising direction worth exploring.

## Chapter 7: Decoding the Trace of Deception

> **Deeply modeling the structure and semantics of spoofing to enhance model discrimination**

As Face Anti-Spoofing (FAS) models face the dual challenges of interpretability and generalization, researchers have begun to focus on the concept of the "spoof trace" — subtle patterns left by fake faces in images, such as color deviations, edge contours, or frequency anomalies.

The five papers in this chapter approach the problem from the perspective of **disentangled representation**, attempting to separate spoof features from genuine facial content, enabling reconstruction, analysis, and even synthesis of spoof samples, allowing models to truly learn to "see through the disguise."

31. [**[20.03] Searching Central Difference Convolutional Networks for Face Anti-Spoofing**](https://arxiv.org/abs/2003.04092)
    Proposes the Central Difference Convolution (CDC) method: by manually defining the hypothesis that "spoof traces should leave differences in local gradients," it disentangles gradient signals of real faces from potential spoofs. Combined with a multiscale attention module, it achieves efficient deployment and strong cross-dataset generalization in face anti-spoofing (FAS) tasks. This work has received a high number of citations.

    :::info
    **Paper Notes**: [**[20.03] CDCN: Between Truth and Falsehood**](https://docsaid.org/en/papers/face-antispoofing/cdcn)
    :::

32. [**[20.07] On Disentangling Spoof Trace for Generic Face Anti-Spoofing**](https://arxiv.org/abs/2007.09273)
    Proposes a multi-scale spoof trace separation model that treats spoof signals as combinations of multi-layer patterns. Through adversarial learning, it reconstructs genuine faces and spoof masks, which can be used to synthesize new attack samples. This work is a representative example of spoof-aware representation learning.

    :::info
    **Paper Notes**: [**[20.07] STDN: Traces of Disguise**](https://docsaid.org/en/papers/face-antispoofing/stdn)
    :::

33. [**[20.08] Face Anti-Spoofing via Disentangled Representation Learning**](https://arxiv.org/abs/2008.08250)
    Decomposes facial features into two subspaces: liveness and identity. Using a CNN architecture to separate low-level and high-level signals, it builds a more transferable live-face classifier, improving stability across different attack types.

    :::info
    **Paper Notes**: [**[20.08] Disentangle-FAS: Untangling the Soul Knot**](https://docsaid.org/en/papers/face-antispoofing/disentangle-fas)
    :::

34. [**[21.10] Disentangled representation with dual-stage feature learning for face anti-spoofing**](https://arxiv.org/abs/2110.09157)
    Employs a dual-stage disentanglement training mechanism to separate facial images into two subspaces—related and unrelated to liveness—and effectively enhances the model’s ability to recognize unseen attack types. This is a key design to strengthen generalization performance.

    :::info
    **Paper Notes**: [**[21.10] DualStage: The Technique of Dual Disentanglement**](https://docsaid.org/en/papers/face-antispoofing/dualstage)
    :::

35. [**[21.12] Dual spoof disentanglement generation for face anti-spoofing with depth uncertainty learning**](https://arxiv.org/abs/2112.00568)
    Introduces the DSDG generation framework that factorizes latent representations of identity and attack texture via a Variational Autoencoder (VAE). It can synthesize large-scale diverse spoof images and incorporates a depth uncertainty module to stabilize depth supervision, serving as a paradigm for "generative adversarial spoofing."

    :::info
    **Paper Notes**: [**[21.12] DSDG: The Eve of Illusion Recombination**](https://docsaid.org/en/papers/face-antispoofing/dsdg)
    :::

---

This chapter marks a critical turning point: from detecting liveness → analyzing spoofing → simulating attacks, Face Anti-Spoofing research is gradually progressing toward a next phase that is “generative, interpretable, and controllable.” These approaches not only improve model accuracy but may also inspire future evolutionary paths in attack and defense strategies.

## Chapter 8: The Chaotic Landscape of the Future

> **From CLIP to human perception, the next frontier of FAS**

As single-modal and single-attack-type solutions fail to meet real-world needs, FAS is stepping into higher-level challenges: **physical + digital dual attacks, semantic-driven recognition, and zero-shot generalization in diverse environments**.

These five representative works are the three major development axes for the future of FAS: **fusion recognition, language modeling, and human-centered perception**.

36. [**[23.09] FLIP: Cross-domain Face Anti-Spoofing with Language Guidance**](https://arxiv.org/abs/2309.16649)
    Applies the CLIP model to the FAS task, guiding visual representation spaces through natural language descriptions to improve cross-domain generalization. The paper proposes semantic alignment and multimodal contrastive learning strategies, achieving true zero-shot FAS under language guidance.

    :::info
    **Paper Notes**: [**[23.09] FLIP: The Defense Spell**](https://docsaid.org/en/papers/face-antispoofing/flip)
    :::

37. [**[24.04] Joint Physical-Digital Facial Attack Detection via Simulating Spoofing Clues**](https://arxiv.org/abs/2404.08450)
    Proposes SPSC and SDSC data augmentation strategies to simulate both physical and digital attack clues, enabling a single model to learn to recognize both types of attacks. This won the CVPR 2024 competition, setting a new paradigm for fusion models.

    :::info
    **Paper Notes**: [**[24.04] PD-FAS: The Illusionary Arena**](https://docsaid.org/en/papers/face-antispoofing/pd-fas)
    :::

38. [**[24.04] Unified Physical-Digital Attack Detection Challenge**](https://arxiv.org/abs/2404.06211)
    Launched the first unified attack detection challenge, releasing the 28,000-entry UniAttackData complex attack dataset and analyzing model architectures, catalyzing the research community toward Unified Attack Detection.

    :::info
    **Paper Notes**: [**[24.04] FAS-Challenge: Arsenal**](https://docsaid.org/en/papers/face-antispoofing/fas-challenge)
    :::

39. [**[24.08] La-SoftMoE CLIP for Unified Physical-Digital Face Attack Detection**](https://arxiv.org/abs/2408.12793)
    Combines CLIP with the Mixture of Experts architecture, introducing a soft-adaptive mechanism to dynamically assign sub-models for complex decision boundaries, providing an efficient parameter selection solution for physical and digital attack fusion handling.

    :::info
    **Paper Notes**: [**[24.08] La-SoftMoE: Sparse Cracks**](https://docsaid.org/en/papers/face-antispoofing/la-softmoe)
    :::

40. [**[25.01] Interpretable Face Anti-Spoofing: Enhancing Generalization with Multimodal Large Language Models**](https://arxiv.org/abs/2501.01720)
    Proposes a novel architecture I-FAS that integrates multimodal large language models, transforming face anti-spoofing into an interpretable visual question answering task. Through semantic annotation, an asymmetric language loss, and a globally aware connector, it significantly improves cross-domain generalization and reasoning capabilities of the model.

    :::info
    **Paper Notes**: [**[25.01] I-FAS: The Final Chapter of Classification**](https://docsaid.org/en/papers/face-antispoofing/i-fas)
    :::

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
