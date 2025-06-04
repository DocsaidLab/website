---
title: "[24.10] FM-CLIP"
authors: Z. Yuan
---

## 來自語言的指引

[**FM-CLIP: Flexible Modal CLIP for Face Anti-Spoofing**](https://dl.acm.org/doi/pdf/10.1145/3664647.3680856)

---

在 FAS 領域中所提到的「多模態」，通常指的是 RGB、深度、紅外等不同的感測器。

但近年來，還有另外一種「模態」大行其道：自然語言。

## 定義問題

Face Anti-Spoofing（FAS）最初的戰場是影像本身。

研究者們設計卷積網路，提取紋理、深度、反射率等特徵，以分辨真假臉孔。

然而隨著攻擊手法演化，高解析度的列印、重播、3D 面具逐漸讓單一模態的防線支離破碎。

為了應對這種攻防升級，FAS 社群引入了多模態融合。RGB 捕捉顏色，IR 感測熱源，Depth 測量結構，在不同感測器的交錯訊號中，試圖拼湊出一幅更接近真實的圖景。

但這條路線也有它的裂痕。

多模態融合倚賴所有模態在訓練與測試階段皆齊備，一旦其中任一感測器資料缺失，系統的辨識能力便如潰堤一般，近乎完全失效。部署上的硬體成本、場景條件的變異，使得「模態一致性」成為一種奢望。

Flexible Modal 的概念應運而生。

它的目標是設計一個模型，能夠在訓練階段學習多模態的特徵，但在測試階段，卻不再依賴所有模態的存在。然而在過去的研究中，Flexible Modal 的設計仍然是針對傳統的模態進行，例如光譜、熱場、幾何訊號等。

自然語言的興起，讓我們看到了另一種可能性。

語言不是直接捕捉世界的光與形，而是對經驗的描述、詮釋與對齊。

它提供了一種超越感測器層級的對齊機制，讓異質的觀測，在語意層次上尋找共通性。

或許，我們能夠在這些破碎模態的縫隙中，以自然語言為橋，重建對真偽的認知。

## 解決問題

:::tip
這篇論文以 CLIP 作為基礎架構，如果你對 CLIP 不熟悉，可以參考我們之前的論文。

- [**[21.03] CLIP: 打碎次元的屏障**](../../multimodality/2103-clip/index.md)

:::

### 模型架構

<div align="center">
<figure style={{"width": "90%"}}>
![model_arch](./img/img1.jpg)
</figure>
</div>

作者提出了 **FM-CLIP**，一個專為 FAS 設計，並基於 CLIP 的多模態對齊模型。

整個架構建立在一個凍結的 CLIP 模型之上。

如上圖所示，FM-CLIP 可以分成兩個主要分支：

- **視覺分支（Visual Branch）**：接收 RGB、Depth 等感測器資料，經過 ViT 影像編碼器處理。
- **語言分支（Language Branch）**：以 prompt learning 生成的文字向量作為輔助訊號，引導視覺特徵對齊。

接下來，我們依照訊號流動的順序，仔細看一下每個部件的設計。

### CMS-Enhancer

ViT 原本是純粹的自注意力網路，缺乏對局部結構與頻率訊號的敏感度。

為了彌補這個缺陷，作者在每個 ViT stage 插入了跨模態仿冒強化模組：

- **Cross-Modal Spoofing Enhancer（CMS-Enhancer）**

將輸入特徵同時分解成兩個平行通道：

- **空間特徵**：使用 Spatial Extractor（SE）進行細粒度紋理提取。
- **頻率特徵**：使用 Frequency Extractor（FE）將影像映射到頻域，抽取高層次結構差異。

**Spatial Extractor (SE)** 的操作方式如下：

$$
F_{\text{SE\_output}}^{(j)} = \text{Conv1}(\text{GELU}(\text{Conv3}(\text{GELU}(\text{Conv1}(F_{\text{input}}^{(j)})))))
$$

也就是一個簡單的卷積結構，搭配 $GELU$ 的激活函數，取得局部影像特徵。

最終加上殘差連接：

$$
\hat{F}_{\text{spatial}}^{(j)} = F_{\text{SE\_output}}^{(j)} \oplus F_{\text{input}}^{(j)}
$$

**Frequency Extractor (FE)** 則是：

$$
F_{\text{FE\_output}}^{(j)} = \sigma(\text{Conv1}(\text{GELU}(\text{Conv1}(\text{DCT}(F_{\text{input}}^{(j)})))))
$$

把影像轉成頻譜圖之後，同樣套上卷積與 $GELU$ 函數，雖然操作類似，但是目標已經改成頻譜圖，因此找出來的特徵可以說是截然不同。

最後輸出和原本的輸入進行點乘計算，用來強化或弱化某個頻率。

$$
\hat{F}_{\text{frequency}}^{(j)} = F_{\text{FE\_output}}^{(j)} \otimes F_{\text{input}}^{(j)}
$$

### Cross-Modal Interactor

不同模態在空間特徵上或許千差萬別，但在頻率空間裡，它們可以被映射到一個共享的中介平面。為了促進這種頻域互動，作者接著設計了**Cross-Modal Interactor (CMI)** 模組：

- 先為每個模態計算一組 gate map，標記出資訊密度高與低的區域。
- 再根據 gate map，從另一個模態補充有用訊息，修補本模態的薄弱區域。

計算 gate map：

$$
M_{\text{freq.RGB}} = \sigma(\text{Conv3}(F_{\text{freq.RGB}}))
$$

$$
M_{\text{freq.Depth}} = \sigma(\text{Conv3}(F_{\text{freq.Depth}}))
$$

這個過程中，由於輸出經過 sigmoid 函數，因此是一個 0~1 之間的數值，意義在於可以保留或關閉模型認為不需要的影像區域。

接著是交互補充過程：

$$
eF_{\text{freq.RGB-Depth}} = (1-M_{\text{freq.RGB}}) \otimes eF_{\text{freq.Depth}}
$$

$$
eF_{\text{freq.Depth-RGB}} = (1-M_{\text{freq.Depth}}) \otimes eF_{\text{freq.RGB}}
$$

意思就是說，假設在原本 RGB 特徵中，模型已經決定保留某些區塊；接著把這個區塊套用到另外一個模態的特徵上，讓模型也特別看看這個「隔壁」家的小孩，有沒有什麼特別的地方。

最後，將本模態原始特徵、加強特徵與補充特徵進行融合：

$$
F_{E\_\text{freq.RGB}} = F_{\text{freq.RGB}} \oplus eF_{\text{freq.RGB}} \oplus eF_{\text{freq.RGB-Depth}}
$$

$$
F_{E\_\text{freq.Depth}} = F_{\text{freq.Depth}} \oplus eF_{\text{freq.Depth}} \oplus eF_{\text{freq.Depth-RGB}}
$$

並且和對應的空間特徵合併，形成增強特徵。

這樣，視覺分支在每一個 ViT block 裡，不只學到自己模態的細節，還吸收了來自其他模態的頻率補充訊息。

### Language-Guided Patch Alignment

視覺訊號的處理告一段落後，作者這裡引入自然語言模態，進一步引導每個 patch 聚焦於仿冒線索。

在文本分支中，作者使用 **Prompt Learning** 技術，初始化一組可學習的 context 向量 $\mathbf{v} = \{v_1, v_2, ..., v_M\}$，並結合類別標籤 $c_i$，形成 prompt：

$$
t_i = \{v_1, v_2, ..., v_M, c_i\}
$$

這個技術本身沒有什麼創新的地方，近幾年只要是想要調用大模型的能力時，大多會採用這個方法，簡單有效。真的要說缺點的話，大概就是學習出來的 Token 本身比較難以解釋。

經過文字編碼器 $g(\cdot)$ 後，生成文本特徵 $f_{\text{text}}$。

在剛才的視覺分支中，我們已經得到了 CLS token $f_{\text{img}}^{(0)}$ 和 Patch tokens $f_{\text{img}}^{(1:N)}$。

作者在這裡採用雙重對齊：

1. **CLS token 對齊**：計算 CLS 與 EOS（real/fake）向量的相似度，用來進行全局分類。
2. **Patch token 對齊（LGPA）**：計算每個 Patch token 和文字特徵的相似度矩陣：

$$
S = f_{\text{img}}^{(1:N)} \cdot (f_{\text{text}})^T
$$

然後進行加權融合：

$$
\hat{f}_{\text{img}}^{(1:N)} = \text{softmax}(S) \cdot f_{\text{text}} + f_{\text{img}}^{(1:N)}
$$

這樣，每一個 Patch 都能根據語言引導，重新聚焦於可能存在仿冒痕跡的局部線索。

### 損失函數設計

為了同時監督全局與局部的對齊，最後作者引入兩個損失項：

- **CLS Loss（全局對齊）**：

  $$
  L_C = \text{CrossEntropy}(p_{\text{cls\_token}}, y)
  $$

- **Patch Loss（局部對齊）**：

  $$
  L_P = \text{CrossEntropy}(p_{\text{patch\_token}}, y)
  $$

最終總損失為：

$$
L_{\text{total}} = L_C + L_P
$$

這種設計讓模型在全局辨識與局部細節間保持張力，既能捕捉宏觀語義，也能聚焦微觀破綻。

## 討論

作者選用三個多模態 FAS 常用資料集進行測試：

- **CASIA-SURF (SURF)**：三模態資料，主攻未知攻擊類型。
- **CASIA-SURF CeFA (CeFA)**：包含種族與模態變異，選用 Protocol 1、2、4。
- **WMCA**：高仿真多攻擊場景，涵蓋「seen」與「unseen」兩種評估情境。

實驗涵蓋兩種測試設定：

- **固定模態（Fixed Modal）**：訓練與測試模態一致。
- **靈活模態（Flexible Modal）**：測試階段僅提供任一單模態資料。

指標方面，使用 APCER、BPCER 與 ACER 作為標準。

### 固定模態結果

:::tip
受限於版面，這裡只放 SURF 的圖表，另外兩個資料集的圖表可以參考原始論文。
:::

<div align="center">
<figure style={{"width": "90%"}}>
![fixed_modal](./img/img2.jpg)
</figure>
</div>

在固定模態情境下，FM-CLIP 顯示出穩定的提升趨勢。

- **SURF 資料集**：
  引入 CMS-Enhancer 後，ACER 由 0.45 降至 0.44；整合 LGPA 後，進一步降至 0.43。
- **WMCA 資料集（unseen protocol）**：
  CMS-Enhancer 使 ACER 從 2.49% 下降至 2.36%；加上 LGPA，FM-CLIP 最終降至 2.29%。
- **CeFA 資料集**：
  在三個 protocol 上，FM-CLIP 均小幅降低 APCER、BPCER、ACER 指標，展現出穩健的跨域泛化能力。

由於 FM-CLIP 的可訓練參數量比 FM-ViT 少，在 WMCA「seen」情境下的絕對表現略低於 FM-ViT，屬可預期的權衡結果。

### 靈活模態結果

![flexible_modal](./img/img3.jpg)

在更具挑戰性的靈活模態測試中，FM-CLIP 展現出明顯優勢。

- **SURF 資料集**：
  在 RGB、Depth、IR 三個單模態下，FM-CLIP 全面超越 FM-ViT，最高達到 2.17% 的 ACER 降幅。
- **CeFA Protocol 4**：
  特別是在 IR 模態，FM-CLIP 相較 FM-ViT 減少了 8.1 的 ACER 指標，顯示對於難以辨識的紅外資料特別有效。
- **WMCA（seen protocol）**：
  FM-CLIP 在所有模態（RGB、Depth、IR）上均有額外提升，並保持穩定的低誤差率。

### 核心組件分析

<div align="center">
<figure style={{"width": "70%"}}>
![core_components](./img/img4.jpg)
</figure>
</div>

作者針對 FM-CLIP 的兩個主要模組，CMS-Enhancer 與 VLA（Vision-Language Alignment），進行了消融實驗。
實驗場景以 WMCA（seen）、SURF、CeFA (Prot.4) 資料集，在靈活模態設定下進行。

結果顯示：

- 單獨引入 **CMS-Enhancer**，ACER 平均下降超過 4%，有效提升視覺特徵的穩定性。
- 單獨引入 **VLA**，同樣帶來約 4% 左右的降幅，證明語言引導在局部特徵對齊上的作用。
- **整合兩者後（FM-CLIP）**，在各資料集上 ACER 分別下降 8%~9%，顯示兩個模組具有互補性。

## 結論

將 VLM 引入 FAS 領域，已經是近年來的熱門趨勢。

在資料來源異質、攻擊方式多變的背景下，依靠單一感測器或手工特徵設計，已愈來愈難支撐穩定的辨識系統。自然語言作為一種高層次對齊機制，提供了跨感測器、跨攻擊型態的潛在連結，也成為許多研究者試圖借力的方向。

基於這篇研究，我們可以看出當前研究的兩個重要方向：

1. **當物理層觀測不可避免地碎片化時，語義層的對齊與修復，將成為辨識系統的重要支柱。**
2. **單純的語言引導尚不足以完全取代感測器層次的資訊補強，頻率空間、局部結構與語義關聯，仍需更緊密地編織在一起。**

FM-CLIP 在這條探索路上，以輕量設計展現了異模態對齊的可行性，也留下了更深層結構建模與主動感知修復的想像空間。
