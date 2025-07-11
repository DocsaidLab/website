---
slug: fas-paper-roadmap
title: Face Anti-Spoofing 技術地圖
authors: Z. Yuan
image: /img/2025/0401.jpg
tags: [face-anti-spoofing, liveness-detection]
description: FAS 的 40 篇論文導讀。
---

Face Anti-Spoofing 是什麼？為什麼它重要？我該怎麼入門？

這篇文章是我閱讀大量文獻後，為正在學習、研究、或開發 FAS 系統的你所整理的完整導讀地圖。

我挑出最具代表性的 40 篇論文，依照時間與技術發展劃分為八大主題，每一篇都會告訴你該讀的理由、關鍵貢獻與適合定位。從傳統 LBP、rPPG、CNN 到 Transformer、CLIP、Vision-Language Model，全部一次掌握。

後續我會在「論文筆記」中分享每篇論文的細節，現在讓我們先掌握整體脈絡。

<!-- truncate -->

## 第一章：起源的低解析光

> **從傳統特徵工程到深度學習的第一道曙光**

Face Anti-Spoofing 的早期研究主要建立在傳統影像處理技術之上，研究者多仰賴紋理、對比、頻率等手工特徵來描述人臉的真實性，並透過經典分類器進行二元辨識。

1. [**[10.09] Face Liveness Detection from a Single Image with Sparse Low Rank Bilinear Discriminative Model**](https://parnec.nuaa.edu.cn/_upload/article/files/4d/43/8a227f2c46bda4c20da97715f010/db1eef47-b25f-4af9-88d4-a8afeccda889.pdf)
   利用 Lambertian 模型與稀疏低秩表示建構特徵空間，有效分離真臉與照片，為早期單張影像活體檢測提供理論與實作依據。

   :::info
   **論文筆記**：[**[10.09] SLRBD: 沈默的反射光**](https://docsaid.org/papers/face-antispoofing/slrbd/)
   :::

2. [**[12.09] On the Effectiveness of Local Binary Patterns in Face Anti-Spoofing**](https://ieeexplore.ieee.org/document/6313548)
   使用 LBP 與其變種特徵，針對平面照片與螢幕重播攻擊進行辨識，並建立 REPLAY-ATTACK 資料集，是最早公開資料集與經典 baseline 組合之一。

   :::info
   **論文筆記**：[**[12.09] LBP: 輕快的微紋理**](https://docsaid.org/papers/face-antispoofing/lbp/)
   :::

3. [**[14.05] Spoofing Face Recognition with 3D Masks**](https://ieeexplore.ieee.org/document/6810829)
   系統性分析 3D 假面對不同臉部辨識系統（2D/2.5D/3D）的攻擊效果，指出傳統對平面假臉的假設在 3D 印製技術下已不再成立。

   :::info
   **論文筆記**：[**[14.05] 3DMAD: 真實的假面**](https://docsaid.org/papers/face-antispoofing/three-d-mad/)
   :::

4. [**[19.09] Biometric Face Presentation Attack Detection with Multi-Channel Convolutional Neural Network**](https://arxiv.org/abs/1909.08848)
   提出多通道 CNN 架構，結合 RGB、深度、紅外與熱感訊號進行辨識，並釋出 WMCA 資料集，提升對高階假臉（如矽膠面具）的偵測能力。

   :::info
   **論文筆記**：[**[19.09] WMCA: 看不見的臉**](https://docsaid.org/papers/face-antispoofing/wmca/)
   :::

5. [**[22.10] Deep Learning for Face Anti-Spoofing: A Survey**](https://ieeexplore.ieee.org/abstract/document/9925105)
   為 FAS 領域第一篇以深度學習為主軸的系統性綜述，涵蓋 pixel-wise 監督、多模態感測器與 domain generalization 等新趨勢，建立知識全景。

   :::info
   **論文筆記**：[**[22.10] FAS Survey: 攻與防的編年史**](https://docsaid.org/papers/face-antispoofing/fas-survey/)
   :::

---

這些方法雖簡單，但奠定了辨識平面假臉（如照片與螢幕重播）的基礎認知，也為後來深度學習技術的導入打下概念框架。

## 第二章：真實世界的舞台

> **FAS 技術從實驗室走向真實場景的里程碑**

資料集與 benchmark 決定了一個領域能否穩定成長。

FAS 技術從單一場景走向多設備、多光源、多攻擊手法，是透過這些具代表性的公開資料集推動而來。

6. [**[17.06] OULU-NPU: A Mobile Face Presentation Attack Database with Real-World Variations**](https://ieeexplore.ieee.org/document/7961798)
   針對手機場景設計的 FAS 資料集，涵蓋裝置、環境光與攻擊手法等多種變因，並設計四種測試協定，成為「泛化能力」評估的里程碑。

   :::info
   **論文筆記**：[**[17.06] OULU-NPU: 四道關卡**](https://docsaid.org/papers/face-antispoofing/oulu-npu/)
   :::

7. [**[20.03] CASIA-SURF CeFA: A Benchmark for Multi-modal Cross-ethnicity Face Anti-Spoofing**](https://arxiv.org/abs/2003.05136)
   全球首個具有「族群標註」的大型多模態 FAS 資料集，涵蓋 RGB、Depth、IR 與多種攻擊類型，特別用於研究族群偏差與模態融合策略。

   :::info
   **論文筆記**：[**[20.03] CeFA: 模型的歧視**](https://docsaid.org/papers/face-antispoofing/cefa/)
   :::

8. [**[20.07] CelebASpoof: Large-scale Face Anti-Spoofing Dataset with Rich Annotations**](https://arxiv.org/abs/2007.12342)
   目前最大規模的 FAS 資料集，超過 62 萬張影像，並含 10 類 spoof 標註與原始 CelebA 的 40 個屬性，可進行多任務與 spoof trace 學習。

   :::info
   **論文筆記**：[**[20.07] CelebA-Spoof: 大規模防偽試煉**](https://docsaid.org/papers/face-antispoofing/celeba-spoof/)
   :::

9. [**[22.01] A Personalized Benchmark for Face Anti-Spoofing**](https://openaccess.thecvf.com/content/WACV2022W/MAP-A/html/Belli_A_Personalized_Benchmark_for_Face_Anti-Spoofing_WACVW_2022_paper.html)
   主張將使用者註冊時的活體影像納入辨識流程，提出 CelebA-Spoof-Enroll 與 SiW-Enroll 兩個新測試配置，探索個人化 FAS 系統的可能性。

   :::info
   **論文筆記**：[**[22.01] Personalized-FAS: 個人化的嘗試**](https://docsaid.org/papers/face-antispoofing/personalized-fas/)
   :::

10. [**[24.02] SHIELD: An Evaluation Benchmark for Face Spoofing and Forgery Detection with Multimodal Large Language Models**](https://arxiv.org/abs/2402.04178)
    結合 LLM 與多模態輸入，提出以 QA 任務形式評估 MLLM 在 spoof/forgery 檢測的推理能力，開啟「以語言建模理解攻擊」的新場域。

    :::info
    **論文筆記**：[**[24.02] SHIELD: 告訴我，為什麼？**](https://docsaid.org/papers/face-antispoofing/shield/)
    :::

## 第三章：跨域的修羅場

> **從單一資料學習到多場景部署的核心技術**

Face Anti-Spoofing 最棘手的問題之一是泛化能力：如何讓模型不只在訓練資料上有效，也能應對新裝置、新環境與新攻擊。

11. [**[20.04] Single-Side Domain Generalization for Face Anti-Spoofing**](https://arxiv.org/abs/2004.14043)
    提出單邊對抗學習策略，只對真臉進行跨域對齊，讓假臉特徵在不同 domain 中自然分散，避免過度壓縮錯誤資訊，是 DG 設計上極具啟發性的方向。

    :::info
    **論文筆記**：[**[20.04] SSDG: 穩定的真實**](https://docsaid.org/papers/face-antispoofing/ssdg/)
    :::

12. [**[21.05] Generalizable Representation Learning for Mixture Domain Face Anti-Spoofing**](https://arxiv.org/abs/2105.02453)
    不假設已知 domain label，而是透過 instance normalization 與 MMD 做無監督聚類與對齊，實現不依賴人工分群的泛化訓練流程。

    :::info
    **論文筆記**：[**[21.05] D²AM: 千域鍛魂術**](https://docsaid.org/papers/face-antispoofing/d2am/)
    :::

13. [**[23.03] Rethinking Domain Generalization for Face Anti-Spoofing: Separability and Alignment**](https://arxiv.org/abs/2303.13662)
    提出 SA-FAS 框架，強調在不同 domain 保留 feature separability，同時讓 live→spoof 的轉變軌跡在各 domain 中一致，是 IRM 理論在 FAS 上的深度應用。

    :::info
    **論文筆記**：[**[23.03] SA-FAS: 超平面的律令**](https://docsaid.org/papers/face-antispoofing/sa-fas/)
    :::

14. [**[24.02] Suppress and Rebalance: Towards Generalized Multi-Modal Face Anti-Spoofing**](https://arxiv.org/abs/2402.19298)
    對多模態 DG 問題進行深入剖析，透過 U-Adapter 壓制不穩定模態的干擾，搭配 ReGrad 動態調節各模態收斂速度，是模態不均與可靠性問題的完整解法。

    :::info
    **論文筆記**：[**[24.02] MMDG: 信任管理學**](https://docsaid.org/papers/face-antispoofing/mmdg/)
    :::

15. [**[24.03] CFPL-FAS: Class Free Prompt Learning for Generalizable Face Anti-spoofing**](https://arxiv.org/abs/2403.14333)
    聚焦於 prompt learning 的手法，強調「無需手動定義類別」的 prompt 設計，屬於一種利用語言提示來協助 FAS 模型泛化的新思路。

    :::info
    **論文筆記**：[**[24.03] CFPL-FAS: 無類別提示學習**](https://docsaid.org/papers/face-antispoofing/cfpl-fas/)
    :::

---

這五篇論文構成了當前 Domain Generalization（DG）主題下的技術主軸，從單邊對抗、無標籤聚類、可分性分析，到融合語言的監督方式，描繪出對跨域挑戰的完整應戰策略。

## 第四章：新世界的崛起

> **從 CNN 到 ViT，FAS 模型的架構革新之路**

Vision Transformer（ViT）的崛起讓影像任務從局部卷積邁入全局建模時代，Face Anti-Spoofing 也不例外。

16. [**[23.01] Domain Invariant Vision Transformer Learning for Face Anti-Spoofing**](https://openaccess.thecvf.com/content/WACV2023/papers/Liao_Domain_Invariant_Vision_Transformer_Learning_for_Face_Anti-Spoofing_WACV_2023_paper.pdf)
    提出 DiVT 架構，透過兩個核心損失函數強化跨域泛化，聚合真臉特徵，形成更一致的 domain-invariant 表徵。實驗顯示，DiVT 在多項 DG-FAS 任務上達成 SOTA 成績，方法精簡卻能有效捕捉跨域辨識的關鍵資訊。

    :::info
    **論文筆記**：[**[23.01] DiVT: 全明星錦標賽**](https://docsaid.org/papers/face-antispoofing/divt/)
    :::

17. [**[23.02] Rethinking Vision Transformer and Masked Autoencoder in Multimodal Face Anti-Spoofing**](https://arxiv.org/abs/2302.05744)
    全面檢討 ViT 在多模態 FAS 中的核心議題，包含輸入設計、預訓練策略與參數微調流程，並提出 AMA adapter 與 M2A2E 預訓練架構，建構跨模態、無標註的自監督流程。

    :::info
    **論文筆記**：[**[23.02] M²A²E: 舉一反三**](https://docsaid.org/papers/face-antispoofing/m2a2e/)
    :::

18. [**[23.04] MA-ViT: Modality-Agnostic Vision Transformers for Face Anti-Spoofing**](https://arxiv.org/abs/2304.07549)
    採單分支 early fusion 架構，透過 Modal-Disentangle Attention 與 Cross-Modal Attention，實現模態不可知的辨識能力，兼顧記憶效率與彈性部署，是 ViT 在實用性上邁出的重要一步。

    :::info
    **論文筆記**：[**[23.04] MA-ViT: 凡所有相，皆是虛妄**](https://docsaid.org/papers/face-antispoofing/ma-vit/)
    :::

19. [**[23.09] S-Adapter: Generalizing Vision Transformer for Face Anti-Spoofing with Statistical Tokens**](https://arxiv.org/abs/2309.04038)
    利用 Efficient Parameter Transfer Learning 架構，在 ViT 上插入 statistical adapters 並固定主網參數，搭配 Token Style Regularization 抑制風格差異，是專為 cross-domain FAS 設計的輕量方案。

    :::info
    **論文筆記**：[**[23.09] S-Adapter: 真實筆記本**](https://docsaid.org/papers/face-antispoofing/s-adapter/)
    :::

20. [**[24.10] FM-CLIP: Flexible Modal CLIP for Face Anti-Spoofing**](https://dl.acm.org/doi/pdf/10.1145/3664647.3680856)
    透過跨模態頻率提取 (CMS-Enhancer) 與文字引導 (LGPA) 動態對齊假臉線索，能於多模態訓練、單一或多模態測試中維持高偵測準確度，在多組資料集上均展現優異的泛化能力。

    :::info
    **論文筆記**：[**[24.10] FM-CLIP: 來自語言的指引**](https://docsaid.org/papers/face-antispoofing/fm-clip/)
    :::

---

這一階段的五篇論文展示了 Transformer 架構如何處理多模態輸入、模態缺失、跨域風格與 local patch 表徵等關鍵挑戰，代表 FAS 模型設計邏輯的全面轉變。

## 第五章：風格之戰

> **當 spoof 來自不同世界，如何打造風格不敏感模型？**

FAS 模型的泛化不只受到 domain shift 的挑戰，更受到不同風格（style）間資訊不對稱的干擾。

這一章聚焦於風格解耦、對抗學習、測試時自適應（test-time adaptation）與 instance-aware 設計，這些方法嘗試讓模型能在未知風格與樣本分布下，依然保持穩定的辨識性能。

21. [**[22.03] Domain Generalization via Shuffled Style Assembly for Face Anti-Spoofing**](https://arxiv.org/abs/2203.05340)
    採內容與風格分離策略，重組風格空間來模擬 style shift，搭配對比學習強調活體相關風格，是 style-aware DG 設計的重要突破。

    :::info
    **論文筆記**：[**[22.03] SSAN: 風格的殘影**](https://docsaid.org/papers/face-antispoofing/ssan/)
    :::

22. [**[22.12] Cyclically Disentangled Feature Translation for Face Anti-spoofing**](https://arxiv.org/abs/2212.03651)
    提出 CDFTN，透過對抗式學習將活體與風格成分分離，生成結合真實標籤與目標域樣貌的 pseudo-labeled 样本，顯著提升跨域偽裝辨識的準確度與穩健性。

    :::info
    **論文筆記**：[**[22.12] CDFTN: 風格的糾纏**](https://docsaid.org/papers/face-antispoofing/cdftn/)
    :::

23. [**[23.04] Instance-Aware Domain Generalization for Face Anti-Spoofing**](https://arxiv.org/abs/2304.05640)
    拋棄粗略的 domain label，改採 instance-level 的風格對齊策略，透過不對稱 whitening、風格增強與動態 kernel 設計，提煉出對風格不敏感的辨識特徵。

    :::info
    **論文筆記**：[**[23.04] IADG: 風格的獨白**](https://docsaid.org/papers/face-antispoofing/iadg/)
    :::

24. [**[23.10] Towards Unsupervised Domain Generalization for Face Anti-Spoofing**](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Towards_Unsupervised_Domain_Generalization_for_Face_Anti-Spoofing_ICCV_2023_paper.html)
    將 unlabeled 資料納入學習流程，透過分割重組與跨域相似度尋找機制，提煉出能適應多種未標註場景的泛化表徵，達成真正無監督的 DG FAS。

    :::info
    **論文筆記**：[**[23.10] UDG-FAS: 風格的碎片**](https://docsaid.org/papers/face-antispoofing/udg-fas/)
    :::

25. [**[23.11] Test-Time Adaptation for Robust Face Anti-Spoofing**](https://papers.bmvc2023.org/0379.pdf)
    在推理階段針對新場景動態調整模型，結合 activation-based pseudo-labeling 與對比學習防止遺忘，使預先訓練的 FAS 模型能在測試時自我優化，提升對未知攻擊的敏銳度。

    :::info
    **論文筆記**：[**[23.11] 3A-TTA: 荒野求生**](https://docsaid.org/papers/face-antispoofing/three-a-tta/)
    :::

---

這五篇從不同角度挑戰了「風格泛化」這個主題，尤其在 instance-based 與 test-time adaptation 的嘗試上，逐步接近實際應用場景的需求。

## 第六章：多模態的召喚術

> **當圖像不再是唯一，聲音與生理訊號進場了**

在傳統 RGB 模型遇到高仿真攻擊與跨域挑戰的瓶頸時，FAS 社群開始探索非視覺訊號，例如 **rPPG、生理訊號、聲波回音** 等輔助資訊，從「人本訊號」出發，建立更難被偽造的辨識依據。

本章精選五篇橫跨生理信號、3D 幾何與聲學感知的代表作，展示多模態 FAS 技術的潛力與未來性。

26. [**[16.12] Generalized face anti-spoofing by detecting pulse from face videos**](https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICPR-2016/media/files/1223.pdf)
    在早期 FAS 場景，就示範了如何在沒有深度或紅外感測器的條件下，光靠人臉心跳訊號也能辨識假臉，凸顯 rPPG 的潛力。

    :::info
    **論文筆記**：[**[16.12] rPPG: 生命的光斑**](https://docsaid.org/papers/face-antispoofing/rppg)
    :::

27. [**[18.09] Remote Photoplethysmography Correspondence Feature for 3D Mask Face Presentation Attack Detection**](https://dl.acm.org/doi/10.1007/978-3-030-01270-0_34)
    首度提出 CFrPPG（對應式 rPPG）特徵來強化活體訊號擷取，在低光源或攝影機晃動情況下也能準確提取心跳軌跡，對抗 3D 面具攻擊表現優異。

    :::info
    **論文筆記**：[**[18.09] CFrPPG: 心跳的殘響**](https://docsaid.org/papers/face-antispoofing/cfrppg)
    :::

28. [**[19.05] Multi-Modal Face Authentication Using Deep Visual and Acoustic Features**](https://ieeexplore.ieee.org/document/8761776)
    利用智慧型手機內建喇叭與麥克風，發射超音波並解析臉部回音，結合 CNN 提取的圖像特徵，打造不需額外硬體的雙模態安全驗證系統。

    :::info
    **論文筆記**：[**[19.05] VA-FAS: 聲波裡的臉**](https://docsaid.org/papers/face-antispoofing/vafas)
    :::

29. [**[22.08] Beyond the Pixel World: A Novel Acoustic-Based Face Anti-Spoofing System for Smartphones**](https://ieeexplore.ieee.org/document/9868051)
    建立 Echo-Spoof 聲學 FAS 資料集，並設計 Echo-FAS 架構，利用聲波重建 3D 幾何與材料資訊，完全不依賴攝影機，是行動裝置中低成本、高抗性的應用典範。

    :::info
    **論文筆記**：[**[22.08] Echo-FAS: 仿冒的回音**](https://docsaid.org/papers/face-antispoofing/echo-fas)
    :::

30. [**[24.03] AFace: Range-Flexible Anti-Spoofing Face Authentication via Smartphone Acoustic Sensing**](https://dl.acm.org/doi/10.1145/3643510)
    延伸 Echo-FAS 思路，加入 iso-depth 模型與距離自適應演算法，能對抗 3D 列印面具，並根據使用者距離自我調整，是聲波式活體驗證走向實用化的關鍵設計。

    :::info
    **論文筆記**：[**[24.03] AFace: 波動的邊界**](https://docsaid.org/papers/face-antispoofing/aface)
    :::

---

這五篇構成非影像模態在 FAS 領域的重要開端，若想要避開傳統攝影機的限制，這將是值得深究的方向。

## 第七章：拆解假象的軌跡

> **深入建模 spoof 的結構與語義，提升模型判別力**

隨著 FAS 模型邁向可解釋性與泛化能力的雙重挑戰，研究者開始關注「spoof trace」這一概念：即假臉在影像中留下的細微模式，例如顏色偏差、邊緣輪廓或頻率異常。

這一章的五篇論文皆從**表徵解耦**（disentanglement）的角度切入，試圖將 spoof 特徵從人臉內容中分離出來，進而重建、分析、甚至合成 spoof 樣本，讓模型真正學會「看穿偽裝」。

31. [**[20.03] Searching Central Difference Convolutional Networks for Face Anti-Spoofing**](https://arxiv.org/abs/2003.04092)
    提出中心差分（CDC）方法：藉由人工定義出「假象應該在局部梯度中留有差異」的假設，將真實臉與潛在偽裝的梯度訊號分開。再搭配多尺度注意力模組，實現高效部署與跨資料集泛化能力的 FAS 解法，有非常高的引用數。

    :::info
    **論文筆記**：[**[20.03] CDCN: 真與假的錯落之間**](https://docsaid.org/papers/face-antispoofing/cdcn)
    :::

32. [**[20.07] On Disentangling Spoof Trace for Generic Face Anti-Spoofing**](https://arxiv.org/abs/2007.09273)
    提出多尺度 spoof trace 分離模型，將偽裝訊號視為多層圖樣組合，透過對抗學習重建真實臉部與 spoof mask，可應用於合成新攻擊樣本，是 spoof-aware 表徵學習的代表作。

    :::info
    **論文筆記**：[**[20.07] STDN: 偽裝的痕跡**](https://docsaid.org/papers/face-antispoofing/stdn)
    :::

33. [**[20.08] Face Anti-Spoofing via Disentangled Representation Learning**](https://arxiv.org/abs/2008.08250)
    將人臉特徵解構為 liveness 與 identity 兩種子空間，透過 CNN 架構分離低階與高階訊號，建立更具可遷移性的活體分類器，提升在不同攻擊類型上的穩定性。

    :::info
    **論文筆記**：[**[20.08] Disentangle-FAS: 斷開魂結**](https://docsaid.org/papers/face-antispoofing/disentangle-fas)
    :::

34. [**[21.10] Disentangled representation with dual-stage feature learning for face anti-spoofing**](https://arxiv.org/abs/2110.09157)
    透過雙階段解耦訓練機制，將人臉影像分離成與活體相關及無關的兩種子空間，並有效增進模型對未知攻擊類型的辨識能力，是加強泛化性能的關鍵設計。

    :::info
    **論文筆記**：[**[21.10] DualStage: 複解耦之術**](https://docsaid.org/papers/face-antispoofing/dualstage)
    :::

35. [**[21.12] Dual spoof disentanglement generation for face anti-spoofing with depth uncertainty learning**](https://arxiv.org/abs/2112.00568)
    提出 DSDG 生成架構，利用 VAE 進行身份與攻擊紋理的因子化潛在表示，能大規模合成多樣化偽臉影像，並引入深度不確定性模組來穩定深度監督，是「以生成對抗偽裝」的典範之一。

    :::info
    **論文筆記**：[**[21.12] DSDG: 假象重組的前夜**](https://docsaid.org/papers/face-antispoofing/dsdg)
    :::

---

本章指出了一個關鍵轉捩點：從辨識活體 → 分析偽裝 → 模擬攻擊，Face Anti-Spoofing 的研究正逐漸走向「可生成、可解釋、可操控」的下一階段。這些方法不僅提升模型準確率，更可能啟發未來的攻防演化路徑。

## 第八章：未來的混沌之境

> **從 CLIP 到人類知覺，FAS 的下一個邊界**

當單一模態、單一攻擊型態都已難以滿足實戰需求，FAS 正邁入更高層次的挑戰：**物理 + 數位雙重攻擊、語意導向辨識、多樣環境的零樣本泛化**。

這五篇代表作是未來 FAS 的三大發展主軸：**融合辨識、語言建模、與人本感知**。

36. [**[23.09] FLIP: Cross-domain Face Anti-Spoofing with Language Guidance**](https://arxiv.org/abs/2309.16649)
    將 CLIP 模型應用於 FAS 任務，透過自然語言描述導引視覺表徵空間，提升跨 domain 的泛化能力，並提出語義對齊與多模態對比學習策略，達成真正語言引導下的 zero-shot FAS。

    :::info
    **論文筆記**：[**[23.09] FLIP: 防禦咒**](https://docsaid.org/papers/face-antispoofing/flip)
    :::

37. [**[24.04] Joint Physical-Digital Facial Attack Detection via Simulating Spoofing Clues**](https://arxiv.org/abs/2404.08450)
    提出 SPSC 與 SDSC 資料擴增策略，模擬物理與數位攻擊線索，讓單一模型能學習同時辨識兩類攻擊，成功在 CVPR2024 比賽中奪冠，樹立融合式模型新典範。

    :::info
    **論文筆記**：[**[24.04] PD-FAS: 幻象的演武場**](https://docsaid.org/papers/face-antispoofing/pd-fas)
    :::

38. [**[24.04] Unified Physical-Digital Attack Detection Challenge**](https://arxiv.org/abs/2404.06211)
    發起首屆統一攻擊辨識挑戰賽，釋出 2.8 萬筆複合型攻擊資料集 UniAttackData，並分析各隊模型架構，是研究界邁向 Unified Attack Detection 的催化劑。

    :::info
    **論文筆記**：[**[24.04] FAS-Challenge: 兵器譜**](https://docsaid.org/papers/face-antispoofing/fas-challenge)
    :::

39. [**[24.08] La-SoftMoE CLIP for Unified Physical-Digital Face Attack Detection**](https://arxiv.org/abs/2408.12793)
    將 CLIP 與 Mixture of Experts 架構結合，引入 soft-adaptive 機制動態分配子模型以應對複雜決策邊界，為物理與數位攻擊融合處理提供高效參數選擇方案。

    :::info
    **論文筆記**：[**[24.08] La-SoftMoE: 稀疏的裂縫**](https://docsaid.org/papers/face-antispoofing/la-softmoe)
    :::

40. [**[25.01] Interpretable Face Anti-Spoofing: Enhancing Generalization with Multimodal Large Language Models**](https://arxiv.org/abs/2501.01720)
    提出一種結合多模態大型語言模型的全新架構 I-FAS，將人臉活體辨識任務轉化為具可解釋性的視覺問答問題，並透過語意標註、非對稱語言損失與全域感知連結器三項關鍵設計，大幅提升模型的跨域泛化與推理能力。

    :::info
    **論文筆記**：[**[25.01] I-FAS: 分類的終章**](https://docsaid.org/papers/face-antispoofing/i-fas)
    :::

---

這一章標誌著 FAS 領域的未來趨勢：**從辨識假臉 → 推測攻擊類型 → 理解語義 → 結合多模態語言邏輯推理**。研究正從「視覺理解」進化到「語意認知」，而攻擊也正從單一模式演化為複雜混合型。

## 結語

真實世界最不缺的就是惡意，只要人臉辨識的需求存在，防偽的需求就不會停止。

從最初的紋理分析、光影建模，到卷積網路的入場，再到 ViT、CLIP、聲波與人類知覺的加入，FAS 技術不斷擴展其邊界。這幾篇論文不只是經典與趨勢的整理，更是一張跨越數十年技術進化的地圖，串連了過去、現在與未來。

在這張地圖中，我們看見：

- **從單模態到多模態**：不只看畫面，更感測深度、聲音、脈動與材質。
- **從分類到解耦**：不只判斷真假，更試圖理解每一種偽裝的構成方式。
- **從辨識到推理**：不只區分活體，更開始理解語意、材料與語言描述背後的真實。
- **從防禦到生成**：不只是被動防守，也開始主動模擬、重建與干預。

如果你正打算進入這個領域，這份技術導讀不會給你「一套萬用解法」，但它能幫你找到自己的出發點：是著迷於 spoof trace 的可視化？還是想探索 CLIP 如何協助安全辨識？或是對聲波與材料辨識感興趣？

無論你來自哪個背景，FAS 都是一個橫跨影像辨識、生物認證、人因感知、語意推理與跨模態融合的交會點。

這場戰役，還遠遠沒到結束的時候。
