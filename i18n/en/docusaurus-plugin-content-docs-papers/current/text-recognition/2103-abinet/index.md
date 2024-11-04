# [21.03] ABINet

## Thinking more!

[**Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition**](https://arxiv.org/abs/2103.06495)

---

Visual recognition may have reached its limits when it comes to understanding text in challenging scenarios. Researchers are now considering how to enable models to "read" and interpret text more like humans.

## Problem Definition

Previous research has often relied solely on visual features for text recognition. However, when text is blurred, occluded, or distorted, models struggle to recognize it accurately. This is because visual features are compromised, making it difficult for the model to detect text reliably.

To address this limitation, researchers have begun incorporating language models to help models understand text contextually. For instance, if we see a blurred text like “APP?E,” a language model can help us infer that the intended word is likely “APPLE.”

Human reading behavior is generally characterized by:

1. **Autonomy**: Humans can learn from both visual and linguistic cues independently, with language knowledge supporting visual understanding of text.
2. **Bidirectionality**: When text is unclear, humans use surrounding context to fill in the blanks.
3. **Iterative Process**: Humans can continuously refine their understanding through repeated reasoning and correction.

This paper’s authors propose that models should replicate these human reading characteristics:

1. **Autonomy**: The model uses separate vision and language modules to model text independently by its visual and linguistic features.
2. **Bidirectionality**: The authors introduce a “Bidirectional Cloze Network” that learns from context on both sides of the text.
3. **Iterative Process**: The model refines its predictions iteratively, gradually improving its accuracy.

With this approach in mind, the authors have defined the conceptual foundation. The remaining step is to prove its effectiveness through experimentation!

## Solution

The authors propose a new model architecture: **ABINet**, designed to tackle the challenges of scene text recognition.

### Visual Model

The visual model in ABINet functions similarly to traditional network architectures, utilizing ResNet for feature extraction to transform an input image $x$ into a feature representation:

$$
F_b = \tau(R(x)) \in \mathbb{R}^{\frac{H}{4} \times \frac{W}{4} \times C}
$$

where $H$ and $W$ represent the image dimensions, and $C$ is the feature dimension.

The visual features $F_b$ are then converted to character probabilities and transcribed in parallel using a positional attention mechanism:

$$
F_v = \text{softmax}\left(\frac{QK^T}{\sqrt{C}}\right)V
$$

Here's a breakdown of each component in the attention calculation:

- **$Q \in \mathbb{R}^{T \times C}$**: This is the positional encoding of the character sequence, encoding each character's position in the sequence to help the model understand the order of characters. Specifically, $T$ represents the sequence length, or the number of characters to be processed, while $C$ denotes the feature dimension.

- **$K = G(F_b) \in \mathbb{R}^{\frac{HW}{16} \times C}$**: The "key" $K$ is derived from $F_b$ through a function $G(\cdot)$, which in this case is implemented as a mini U-Net structure. This reduces the feature dimension of $F_b$ to match the character sequence length, ensuring consistency between the spatial dimensions of the image features and the character sequence.

- **$V = H(F_b) \in \mathbb{R}^{\frac{HW}{16} \times C}$**: The "value" $V$ is obtained by applying an identity mapping $H(\cdot)$ on $F_b$, meaning that $F_b$ is passed directly without additional transformations. This gives $V$ the same dimensions as $K$, preserving consistency in feature space and spatial resolution.

The purpose of this process is to map the visual features into character probability predictions based on positional information. This enables the model to perform accurate character recognition by leveraging both the sequence order and spatial information from the image.

:::tip
In essence, the positional attention mechanism transforms the image features into corresponding character probabilities, enabling the model to recognize characters accurately by combining their sequential position and image-based features.
:::

### Language Model

The language model in ABINet is integrated after the visual model, using several key strategies to improve text recognition performance:

1. **Autonomous Strategy**:

   As shown in the diagram, the language model functions independently as a spelling correction model. It takes character probability vectors as input and outputs the probability distribution for each character. This independence allows the language model to be trained on unlabeled text data separately, enhancing the model's interpretability and modularity. Additionally, the language model can be replaced or adjusted independently.

   To ensure this autonomy, the authors introduce a technique called Blocking Gradient Flow (BGF), which prevents gradients from the visual model from flowing into the language model, thus preserving the language model's independence.

   This approach enables the model to leverage advancements in natural language processing (NLP). For instance, it allows the use of various pre-trained language models to boost performance as needed.

2. **Bidirectional Strategy**:

   The language model calculates conditional probabilities for bidirectional and unidirectional representations, denoted as $P(y_i | y_n, \dots, y_{i+1}, y_{i-1}, \dots, y_1)$ for bidirectional and $P(y_i | y_{i-1}, \dots, y_1)$ for unidirectional. Bidirectional modeling provides richer semantic information by considering both left and right contexts.

   Similar to the Masked Language Model (MLM) in BERT, which uses a [MASK] token to predict a character $y_i$, direct use of MLM is computationally expensive because each string would need to undergo multiple mask operations to predict each character. To improve efficiency, the authors propose the Bidirectional Cloze Network (BCN), which achieves bidirectional representation without repetitive masking.

   BCN adopts a Transformer decoder-like structure but with unique modifications. Instead of causal masking, BCN uses a custom attention mask to prevent each character from "seeing" itself, thus avoiding information leakage.

   The mask matrix $M$ in BCN’s multi-head attention block is constructed as follows:

   $$
   M_{ij} =
   \begin{cases}
     0, & i \neq j \\
     -\infty, & i = j
   \end{cases}
   $$

   Here, $K_i = V_i = P(y_i) W_l$, where $P(y_i) \in \mathbb{R}^c$ is the probability distribution of character $y_i$, and $W_l \in \mathbb{R}^{c \times C}$ is a linear mapping matrix.

   The multi-head attention computation is given by:

   $$
   F_{\text{mha}} = \text{softmax}\left(\frac{QK^T}{\sqrt{C}} + M\right)V
   $$

   In this, $Q \in \mathbb{R}^{T \times C}$ represents positional encodings in the first layer and outputs from the previous layer in subsequent layers, while $K$ and $V \in \mathbb{R}^{T \times C}$ come from the linear mapping of character probabilities $P(y_i)$.

   The BCN’s cloze-like attention mask enables the model to learn a stronger bidirectional representation, capturing more complete semantic context than unidirectional models.

   :::tip
   Key Points:

   1. BCN uses a decoder structure but without causal masking, enabling parallel decoding.
   2. Instead of [MASK], BCN applies a diagonal mask to improve computational efficiency.
      :::

3. **Iterative Strategy**:

   The authors propose an Iterative Correction strategy to address noise in the visual model’s input, which can lower prediction confidence during Transformer’s parallel decoding. In the first iteration, $y_{i_1}$ represents the visual model's probability predictions. In subsequent iterations, $y_{i \geq 2}$ comes from the previous fusion model's predictions.

   :::tip
   This approach is akin to using multiple Transformer layers, with the authors referring to this as “iterative correction.”
   :::

4. **Fusion**

   Since the visual model is trained on image data and the language model on text data, these two sources of information need to be aligned. The authors employ a "gated mechanism" to fuse the visual and language features effectively, balancing their contributions in the final prediction.

   Visual features $F_v$ and language features $F_l$ are concatenated, and a linear mapping $W_f \in \mathbb{R}^{2C \times C}$ compresses the concatenated features to match the dimensions of $F_v$ and $F_l$.

   The gated vector $G$ is calculated as:

   $$
   G = \sigma([F_v, F_l] W_f)
   $$

   Here, $\sigma$ is the Sigmoid function, which bounds $G$ between 0 and 1, controlling the balance between the visual and language features in the final output.

   The fused feature $F_f$ is obtained by weighted combining $F_v$ and $F_l$ as follows:

   $$
   F_f = G \odot F_v + (1 - G) \odot F_l
   $$

   where $\odot$ represents element-wise multiplication. When $G$ is close to 1, visual features $F_v$ have a stronger influence, while a value close to 0 gives more weight to language features $F_l$.

5. **Supervised Training**

   ABINet is trained end-to-end with a multi-task objective, combining the losses from the visual, language, and fused features.

   The objective function is:

   $$
   L = \lambda_v L_v + \frac{\lambda_l}{M} \sum_{i=1}^{M} L_l^i + \frac{1}{M} \sum_{i=1}^{M} L_f^i
   $$

   where:

   - **$L_v$**: Cross-entropy loss for visual features $F_v$.
   - **$L_l$**: Cross-entropy loss for language features $F_l$.
   - **$L_f$**: Cross-entropy loss for fused features $F_f$.
   - **$L_l^i$** and **$L_f^i$**: Losses at iteration $i$ for language and fused features, respectively.
   - **$\lambda_v$** and **$\lambda_l$**: Balancing factors for adjusting the contributions of each loss term, ensuring a balanced influence from visual and language features in training.

### Implementation Details

- **Training Datasets**: The model is trained on two synthetic datasets, MJSynth (MJ) and SynthText (ST).
- **Testing Datasets**: The evaluation is conducted on six standard benchmark datasets: ICDAR 2013 (IC13), ICDAR 2015 (IC15), IIIT 5K-Words (IIIT), Street View Text (SVT), Street View Text-Perspective (SVTP), and CUTE80 (CUTE).
- **Unlabeled Data for Semi-supervised Learning**: The Uber-Text dataset (with labels removed) is used to assess the effectiveness of semi-supervised learning.
- **Model Configuration**: The model dimension $C$ is set to 512, with the Bidirectional Cloze Network (BCN) comprising 4 layers and 8 attention heads per layer. The balancing factors $\lambda_v$ and $\lambda_l$ are both set to 1.
- **Image Preprocessing**: Images are resized to 32×128 and augmented with techniques such as geometric transformations, image quality degradation, and color jittering to improve robustness.
- **Training Environment**: The training is conducted on four NVIDIA 1080Ti GPUs with a batch size of 384. The ADAM optimizer is used with an initial learning rate of $1 \times 10^{-4}$, which decays to $1 \times 10^{-5}$ after the sixth epoch.

## Discussion

### Comparison with Other Methods

To ensure a fair and rigorous comparison, the authors re-implemented the SOTA algorithm SRN using the same experimental configuration as ABINet. Two re-implemented versions of SRN were tested with different visual models (VMs), alongside several adjustments, such as replacing the VM, removing the side effects of multi-scale training, and applying learning rate decay. These modifications resulted in improved performance compared to the original SRN as reported.

- **Comparison between ABINet-SV and SRN-SV**: ABINet-SV outperformed SRN-SV across multiple datasets.
- **Performance of ABINet-LV**: When paired with a stronger VM, ABINet-LV also demonstrated superior performance.

ABINet trained on MJ and ST performed particularly well on challenging datasets like SVT, SVTP, and IC15, which contain numerous low-quality images with noise and blur. ABINet leveraged linguistic information to significantly enhance recognition accuracy, especially under these difficult conditions.

Furthermore, ABINet was able to recognize irregular fonts and non-standard layouts effectively, thanks to the complementary role of language information in visual feature processing. Even without image rectification, ABINet achieved second-best performance on the CUTE dataset, showing its robust ability to handle irregular text.

### Ablation Study - Visual Model

In this ablation study, the authors mainly compared different feature extraction and sequence modeling methods.

The proposed "positional attention" technique demonstrated a stronger capacity for representing key-value vectors compared to common parallel attention methods.

Additionally, upgrading the visual model (VM) significantly improved accuracy, albeit at the cost of additional parameters and computational overhead. Alternatively, performance can be enhanced by using positional attention for feature extraction and employing a deeper Transformer for sequence modeling in the VM.

### Ablation Study - Language Model

- **PVM**: Indicates supervised pre-training of the visual model (VM) on the MJ and ST datasets.
- **$\text{PLM}_{in}$**: Refers to self-supervised pre-training of the language model (LM) on the MJ and ST datasets.
- **$\text{PLM}_{out}$**: Refers to self-supervised pre-training of the language model on the large-scale WikiText-103 dataset.
- **AGF (Allowing Gradient Flow)**: This setting allows gradient flow between the visual model and the language model.

---

**Findings**:

- **PVM**: Supervised pre-training of the VM (PVM) results in an average accuracy improvement of 0.6%-0.7%, showing that pre-training the visual model significantly enhances its text recognition performance.

- **$\text{PLM}_{in}$**: The effect of $\text{PLM}_{in}$ is limited, likely because the MJ and ST datasets lack the diversity and language structure needed to provide comprehensive linguistic information.

- **$\text{PLM}_{out}$**: Pre-training the LM on the larger and more diverse WikiText-103 dataset ($\text{PLM}_{out}$) leads to a significant improvement, as this dataset enriches the LM's understanding of textual context, enhancing performance even when accuracy is already high.

- **AGF**: Enabling AGF results in a 0.9% drop in average accuracy and a sharp decrease in training loss, indicating that the LM may overfit when gradients flow from the VM, as it relies too heavily on visual cues. Blocking Gradient Flow (BGF) helps the LM learn language features independently, thereby improving generalization.

### Ablation Study - Bidirectional Strategy

Since BCN is a Transformer-based variant, it was compared to SRN, another Transformer-based model.

To ensure fairness, the experiments were conducted under identical conditions except for network structure. SV and LV were used as visual models (VMs) to evaluate BCN's effectiveness at different accuracy levels.

- **BCN vs. SRN-U**: BCN, when compared to SRN's unidirectional version (SRN-U), has similar parameters and inference speed but demonstrates a competitive advantage in accuracy across different VMs.

- **BCN vs. Bidirectional SRN**: BCN outperforms the bidirectional SRN, especially on challenging datasets like IC15 and CUTE. Additionally, ABINet with BCN is 20%-25% faster than SRN, making it more practical for large-scale applications.

### Visualization Analysis

To understand how BCN operates within ABINet, the authors visualized the top-5 prediction probabilities for the word "today," as shown in the figure above.

- When the input is “-oday” or “tod-y,” BCN confidently predicts “t” and “a,” contributing positively to the final fusion predictions.
- For erroneous characters like “l” and “o,” BCN’s confidence is low, minimizing their impact on the final prediction.
- When multiple erroneous characters are present, BCN struggles to restore the correct text due to insufficient contextual support.

This visualization demonstrates BCN’s ability to leverage context for accurate predictions and highlights its limitations when handling extensive errors in the input text.

## Conclusion

This paper introduces a language model branch into text recognition models. Through the autonomous, bidirectional, and iterative design, ABINet effectively improves text recognition accuracy and generalization. Experimental results demonstrate that ABINet achieves outstanding performance across multiple benchmark datasets, particularly excelling in irregular text and low-quality images.

:::tip
This approach is reminiscent of early text recognition models, which often used a "dictionary" to refine output predictions by matching them to the closest valid word. However, as models like CRNN became popular, the use of dictionaries declined due to limitations: dictionaries couldn’t be jointly trained with the model, and they constrained the diversity of output due to their fixed size.

In a way, we’ve come full circle, but now the dictionary concept has been replaced with a language model that aids in text interpretation. The significant advantage here is that the language model can be trained jointly with the visual model. However, similar to dictionary-based methods, language models are constrained by the diversity of training data. For instance, they might struggle with texts lacking semantic meaning, such as license plates or serial numbers, where the language model may not be beneficial. Additionally, for text containing puns or phonetic variations, the language model might introduce misinterpretations.

Despite these limitations, integrating language models is a highly promising direction. In the coming years, we can expect a substantial focus in text recognition research on incorporating language models, and it will be fascinating to follow developments in this field.
:::
