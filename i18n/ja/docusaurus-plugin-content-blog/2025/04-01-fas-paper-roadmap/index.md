---
slug: fas-paper-roadmap
title: Face Anti-Spoofing 技術地図
authors: Z. Yuan
image: /ja/img/2025/0401.jpg
tags: [face-anti-spoofing, liveness-detection]
description: FAS の40本の論文ガイド。
---

Face Anti-Spoofing とは何か？ なぜ重要なのか？ どう始めれば良いのか？

この記事は、私が膨大な文献を読んだ後、FAS システムを学び、研究し、開発しているあなたのために整理した完全なガイドマップです。

最も代表的な 40 本の論文を、時系列と技術の発展に基づいて 8 つのテーマに分け、各論文がなぜ読むべきか、重要な貢献と適切なポジショニングについて説明します。伝統的な LBP、rPPG、CNN から、Transformer、CLIP、Vision-Language Model に至るまで、すべて一度に把握できます。

今後、「論文ノート」で各論文の詳細を共有する予定ですが、まずは全体的な流れを把握しましょう。

<!-- truncate -->

## 第一章：起源の低解像度光

> **伝統的な特徴量エンジニアリングから深層学習の最初の光へ**

Face Anti-Spoofing の初期の研究は主に伝統的な画像処理技術に基づいており、研究者はテクスチャ、コントラスト、周波数などの手作業で特徴量を用いて顔の真偽を判断し、古典的な分類器を使って二項分類を行っていました。

1. [**[10.09] Face Liveness Detection from a Single Image with Sparse Low Rank Bilinear Discriminative Model**](https://parnec.nuaa.edu.cn/_upload/article/files/4d/43/8a227f2c46bda4c20da97715f010/db1eef47-b25f-4af9-88d4-a8afeccda889.pdf)
   ランバーティアンモデルと疎低ランク表示を用いて特徴空間を構築し、真顔と写真を効果的に分離。初期の単一画像での活体検出に理論的および実装の基盤を提供しました。

   :::info
   **論文ノート**：[**[10.09] SLRBD: 静かな反射光**](https://docsaid.org/ja/papers/face-antispoofing/slrbd/)
   :::

2. [**[12.09] On the Effectiveness of Local Binary Patterns in Face Anti-Spoofing**](https://ieeexplore.ieee.org/document/6313548)
   LBP とその変種を用いて、平面写真とスクリーン再生攻撃を識別し、REPLAY-ATTACK データセットを構築。最も初期の公開データセットおよび古典的なベースラインの組み合わせの一つです。

   :::info
   **論文ノート**：[**[12.09] LBP: 軽快な微細構造**](https://docsaid.org/ja/papers/face-antispoofing/lbp/)
   :::

3. [**[14.05] Spoofing Face Recognition with 3D Masks**](https://ieeexplore.ieee.org/document/6810829)
   3D マスクが異なる顔認識システム（2D/2.5D/3D）に対する攻撃効果を系統的に分析し、伝統的な平面顔に対する仮定が 3D 印刷技術では成り立たないことを指摘。

   :::info
   **論文ノート**：[**[14.05] 3DMAD: 現実の仮面**](https://docsaid.org/ja/papers/face-antispoofing/three-d-mad/)
   :::

4. [**[19.09] Biometric Face Presentation Attack Detection with Multi-Channel Convolutional Neural Network**](https://arxiv.org/abs/1909.08848)
   RGB、深度、赤外線、熱感知信号を組み合わせた多チャネル CNN アーキテクチャを提案し、WMCA データセットを発表。高次の偽顔（シリコンマスクなど）の検出能力を向上。

   :::info
   **論文ノート**：[**[19.09] WMCA: 見えない顔**](https://docsaid.org/ja/papers/face-antispoofing/wmca/)
   :::

5. [**[22.10] Deep Learning for Face Anti-Spoofing: A Survey**](https://ieeexplore.ieee.org/abstract/document/9925105)
   FAS 分野で初めての深層学習に基づいた系統的なレビュー論文。ピクセル単位の監視、多モーダルセンサー、ドメイン一般化など新しいトレンドを取り上げ、知識の全体像を構築。

   :::info
   **論文ノート**：[**[22.10] FAS Survey: 攻撃と防御の年代記**](https://docsaid.org/ja/papers/face-antispoofing/fas-survey/)
   :::

---

これらの手法は単純であるものの、平面顔（写真やスクリーン再生）の認識基盤を築くものであり、後の深層学習技術導入に向けた概念的枠組みを作り上げました。

## 第二章：現実世界の舞台

> **FAS 技術が実験室から現実のシーンに進出するマイルストーン**

データセットとベンチマークは、ある分野が安定的に成長できるかどうかを決定します。

FAS 技術は、単一のシーンから複数のデバイス、複数の光源、複数の攻撃手法に対応するようになり、これらの代表的な公開データセットによって推進されてきました。

6. [**[17.06] OULU-NPU: A Mobile Face Presentation Attack Database with Real-World Variations**](https://ieeexplore.ieee.org/document/7961798)
   モバイルシーン向けに設計された FAS データセットで、デバイス、環境光、攻撃手法などのさまざまな変数を含み、4 つのテストプロトコルを設計。これにより「一般化能力」の評価が可能になったマイルストーン。

   :::info
   **論文ノート**：[**[17.06] OULU-NPU: 四つの試練**](https://docsaid.org/ja/papers/face-antispoofing/oulu-npu/)
   :::

7. [**[20.03] CASIA-SURF CeFA: A Benchmark for Multi-modal Cross-ethnicity Face Anti-Spoofing**](https://arxiv.org/abs/2003.05136)
   世界初の「民族タグ付け」のある大型多モーダル FAS データセットで、RGB、深度、IR および複数の攻撃タイプを含み、特に民族偏向とモーダル融合戦略の研究に役立つ。

   :::info
   **論文ノート**：[**[20.03] CeFA: モデルの偏見**](https://docsaid.org/ja/papers/face-antispoofing/cefa/)
   :::

8. [**[20.07] CelebASpoof: Large-scale Face Anti-Spoofing Dataset with Rich Annotations**](https://arxiv.org/abs/2007.12342)
   現在最大規模の FAS データセットで、62 万枚以上の画像を含み、10 種類の spoof タグと元の CelebA の 40 の属性が含まれており、多タスクおよび spoof トレース学習に適しています。

   :::info
   **論文ノート**：[**[20.07] CelebA-Spoof: 大規模な偽造防止の試練**](https://docsaid.org/ja/papers/face-antispoofing/celeba-spoof/)
   :::

9. [**[22.01] A Personalized Benchmark for Face Anti-Spoofing**](https://openaccess.thecvf.com/content/WACV2022W/MAP-A/html/Belli_A_Personalized_Benchmark_for_Face_Anti-Spoofing_WACVW_2022_paper.html)
   ユーザー登録時の活体画像を識別プロセスに組み込む提案。CelebA-Spoof-Enroll および SiW-Enroll という 2 つの新しいテスト設定を提案し、個人化 FAS システムの可能性を探る。

   :::info
   **論文ノート**：[**[22.01] Personalized-FAS: 個人化の試み**](https://docsaid.org/ja/papers/face-antispoofing/personalized-fas/)
   :::

10. [**[24.02] SHIELD: An Evaluation Benchmark for Face Spoofing and Forgery Detection with Multimodal Large Language Models**](https://arxiv.org/abs/2402.04178)
    LLM と多モーダル入力を組み合わせ、QA タスク形式で MLLM の spoof/forgery 検出における推論能力を評価。攻撃を「言語モデリングで理解する」という新しい領域を開拓。

    :::info
    **論文ノート**：[**[24.02] SHIELD: 教えてください、なぜ？**](https://docsaid.org/ja/papers/face-antispoofing/shield/)
    :::

## 第三章：跨領域の修羅場

> **単一データ学習から多シーン展開の核心技術**

Face Anti-Spoofing（FAS）の最も難しい問題の一つは、一般化能力です: モデルが訓練データだけでなく、新しいデバイス、新しい環境、新しい攻撃にも対応できるようにする方法。

11. [**[20.04] Single-Side Domain Generalization for Face Anti-Spoofing**](https://arxiv.org/abs/2004.14043)
    単一の対抗学習戦略を提案し、真顔のみでドメイン間調整を行い、偽顔の特徴を異なるドメインで自然に分散させることで、誤った情報の過度な圧縮を避ける。これは DG 設計における非常に示唆に富んだ方向性です。

    :::info
    **論文ノート**：[**[20.04] SSDG: 安定した真実**](https://docsaid.org/ja/papers/face-antispoofing/ssdg/)
    :::

12. [**[21.05] Generalizable Representation Learning for Mixture Domain Face Anti-Spoofing**](https://arxiv.org/abs/2105.02453)
    ドメインラベルを既知とせず、インスタンス正規化と MMD を使用して、無監督のクラスタリングと調整を実現。人工的なクラスタリングに依存しない一般化訓練フローを実現。

    :::info
    **論文ノート**：[**[21.05] D²AM: 千界鍛魂術**](https://docsaid.org/ja/papers/face-antispoofing/d2am/)
    :::

13. [**[23.03] Rethinking Domain Generalization for Face Anti-Spoofing: Separability and Alignment**](https://arxiv.org/abs/2303.13662)
    SA-FAS フレームワークを提案し、異なるドメインで特徴の分離性を保ちながら、live→spoof の変化軌跡が各ドメインで一貫するように強調。これは IRM 理論を FAS に深く適用したものです。

    :::info
    **論文ノート**：[**[23.03] SA-FAS: 超平面の法則**](https://docsaid.org/ja/papers/face-antispoofing/sa-fas/)
    :::

14. [**[24.02] Suppress and Rebalance: Towards Generalized Multi-Modal Face Anti-Spoofing**](https://arxiv.org/abs/2402.19298)
    多モーダル DG 問題を深く分析し、U-Adapter を使用して不安定なモーダルの干渉を抑制し、ReGrad で各モーダルの収束速度を動的に調整することで、モーダル不均衡と信頼性の問題に対する完全な解決策を提供。

    :::info
    **論文ノート**：[**[24.02] MMDG: 信頼管理学**](https://docsaid.org/ja/papers/face-antispoofing/mmdg/)
    :::

15. [**[24.03] CFPL-FAS: Class Free Prompt Learning for Generalizable Face Anti-spoofing**](https://arxiv.org/abs/2403.14333)
    　 プロンプトラーニングの手法に焦点を当てており、「手動でクラスを定義する必要がない」プロンプト設計を強調している。これは、言語プロンプトを活用して FAS モデルの汎化能力を高める新たなアプローチである。

    :::info
    **論文ノート**：[**[24.03] CFPL-FAS: クラスなしのプロンプト学習**](https://docsaid.org/ja/papers/face-antispoofing/cfpl-fas/)
    :::

---

これらの 5 本の論文は、現在の Domain Generalization（DG）テーマの技術的軸を構成しています。単一の対抗、ラベルなしクラスタリング、分離性分析から、言語を統合した監視方法に至るまで、クロスドメインの課題に対する完全な戦略を描き出しています。

## 第四章：新世界の勃興

> **CNN から ViT へ、FAS モデルのアーキテクチャ革新の道**

Vision Transformer（ViT）の登場により、画像タスクは局所的な畳み込みから全体的なモデリング時代へと進化しました。Face Anti-Spoofing も例外ではありません。

16. [**[23.01] Domain Invariant Vision Transformer Learning for Face Anti-Spoofing**](https://openaccess.thecvf.com/content/WACV2023/papers/Liao_Domain_Invariant_Vision_Transformer_Learning_for_Face_Anti-Spoofing_WACV_2023_paper.pdf)
    DiVT アーキテクチャを提案し、2 つの主要な損失関数を通じてクロスドメイン汎化性能を強化。真の顔特徴を集約することで、より一貫性のあるドメイン不変表現を形成する。実験では、DiVT が複数の DG-FAS タスクにおいて SOTA の成果を達成しており、手法は簡潔でありながら、クロスドメイン認識における重要な情報を効果的に捉えることができることが示された。

    :::info
    **論文ノート**：[**[23.01] DiVT: オールスター選手権**](https://docsaid.org/ja/papers/face-antispoofing/divt/)
    :::

17. [**[23.02] Rethinking Vision Transformer and Masked Autoencoder in Multimodal Face Anti-Spoofing**](https://arxiv.org/abs/2302.05744)
    ViT が多モーダル FAS における主要な問題を全面的に再考。入力設計、事前学習戦略、パラメータ微調整フローを含む、AMA アダプターと M2A2E 事前学習アーキテクチャを提案し、クロスモーダルかつラベルなしの自己監督プロセスを構築。

    :::info
    **論文ノート**：[**[23.02] M²A²E: 舉一反三**](https://docsaid.org/ja/papers/face-antispoofing/m2a2e/)
    :::

18. [**[23.04] MA-ViT: Modality-Agnostic Vision Transformers for Face Anti-Spoofing**](https://arxiv.org/abs/2304.07549)
    単一分岐の早期融合アーキテクチャを採用し、Modal-Disentangle Attention と Cross-Modal Attention を通じて、モーダルに依存しない識別能力を実現。記憶効率と柔軟な展開を両立させた、ViT の実用性における重要な一歩。

    :::info
    **論文ノート**：[**[23.04] MA-ViT: 凡所有相，皆是虚妄**](https://docsaid.org/ja/papers/face-antispoofing/ma-vit/)
    :::

19. [**[23.09] S-Adapter: Generalizing Vision Transformer for Face Anti-Spoofing with Statistical Tokens**](https://arxiv.org/abs/2309.04038)
    Efficient Parameter Transfer Learning アーキテクチャを利用して、ViT に統計的アダプターを挿入し、主ネットワークのパラメータを固定。Token Style Regularization でスタイル差を抑制し、クロスドメイン FAS に特化した軽量ソリューション。

    :::info
    **論文ノート**：[**[23.09] S-Adapter: 実際のノート**](https://docsaid.org/ja/papers/face-antispoofing/s-adapter/)
    :::

20. [**[24.10] FM-CLIP: Flexible Modal CLIP for Face Anti-Spoofing**](https://dl.acm.org/doi/pdf/10.1145/3664647.3680856)
    クロスモーダル詐欺強化器（CMS-Enhancer）とテキスト誘導（LGPA）による偽顔の動的アライメントにより、マルチモーダル訓練および単一または複数のモーダルテストで高い検出精度を維持し、複数のデータセットにおいて優れた汎化能力を示します。

    :::info
    **論文ノート**：[**[24.10] FM-CLIP: 言語からの指針**](https://docsaid.org/ja/papers/face-antispoofing/fm-clip/)
    :::

---

この段階の 5 本の論文は、Transformer アーキテクチャが多モーダル入力、モーダル欠損、クロスドメインスタイル、局所パッチ表現などの重要な課題をどのように処理しているかを示しています。これは FAS モデル設計ロジックの全面的な転換を表しています。

## 第五章：スタイルの戦い

> **異なる世界からのスプーフィング、どのようにしてスタイルに敏感でないモデルを作るか？**

FAS モデルの一般化は、ドメインシフトの挑戦だけでなく、異なるスタイル間の情報の非対称性による干渉も受けています。

この章では、スタイルの解消、対抗学習、テスト時適応、インスタンス認識設計に焦点を当てています。これらの手法は、未知のスタイルやサンプル分布のもとでも安定した認識性能を保つことを目指しています。

21. [**[22.03] Domain Generalization via Shuffled Style Assembly for Face Anti-Spoofing**](https://arxiv.org/abs/2203.05340)
    コンテンツとスタイルの分離戦略を採用し、スタイル空間を再構成してスタイルシフトをシミュレートします。コントラスト学習を組み合わせ、生体性に関連するスタイルを強調することで、スタイル認識に基づいたドメイン一般化（DG）設計における重要なブレークスルーを実現します。

    :::info
    **論文ノート**：[**[22.03] SSAN: スタイルの残像**](https://docsaid.org/ja/papers/face-antispoofing/ssan/)
    :::

22. [**[22.12] Cyclically Disentangled Feature Translation for Face Anti-spoofing**](https://arxiv.org/abs/2212.03651)
    CDFTN を提案し、対抗学習によって生体性とスタイル成分を分離し、実際のラベルとターゲットドメインの外観を組み合わせた擬似ラベル付きサンプルを生成します。これにより、クロスドメインでの偽装認識の精度と堅牢性が大幅に向上します。

    :::info
    **論文ノート**：[**[22.12] CDFTN: スタイルの絡み合い**](https://docsaid.org/ja/papers/face-antispoofing/cdftn/)
    :::

23. [**[23.04] Instance-Aware Domain Generalization for Face Anti-Spoofing**](https://arxiv.org/abs/2304.05640)
    粗いドメインラベルを放棄し、インスタンスレベルのスタイルアライメント戦略を採用します。非対称ホワイトニング、スタイル強化、動的カーネル設計を通じて、スタイルに敏感でない認識特徴を洗練させます。

    :::info
    **論文ノート**：[**[23.04] IADG: スタイルの独白**](https://docsaid.org/ja/papers/face-antispoofing/iadg/)
    :::

24. [**[23.10] Towards Unsupervised Domain Generalization for Face Anti-Spoofing**](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Towards_Unsupervised_Domain_Generalization_for_Face_Anti-Spoofing_ICCV_2023_paper.html)
    ラベルのないデータを学習プロセスに取り入れ、分割再構成とクロスドメイン類似度検索機構を使用して、複数のラベルなしシナリオに適応する一般化された表現を抽出します。これにより、真の無監督型ドメイン一般化（DG）FAS が達成されます。

    :::info
    **論文ノート**：[**[23.10] UDG-FAS: スタイルの断片**](https://docsaid.org/ja/papers/face-antispoofing/udg-fas/)
    :::

25. [**[23.11] Test-Time Adaptation for Robust Face Anti-Spoofing**](https://papers.bmvc2023.org/0379.pdf)
    推論段階で新しいシーンに対してモデルを動的に調整し、アクティベーションベースの擬似ラベリングとコントラスト学習を組み合わせて忘却を防止し、事前に学習した FAS モデルがテスト時に自己最適化できるようにし、未知の攻撃に対する感度を向上させます。

    :::info
    **論文ノート**：[**[23.11] 3A-TTA: 荒野でのサバイバル**](https://docsaid.org/ja/papers/face-antispoofing/three-a-tta/)
    :::

---

これらの 5 篇は、異なる観点から「スタイル一般化」というテーマに挑戦しています。特に、インスタンスベースとテスト時適応の試みでは、実際の応用シナリオの要求に徐々に近づいています。

## 第六章：多モーダルの召喚術

> **画像だけではなく、音声や生理信号も登場**

従来の RGB モデルが高精度攻撃やクロスドメインの課題に直面した時、FAS コミュニティは視覚以外の信号、例えば**rPPG、生理信号、音波エコー**などの補助情報を探索し、「人間の信号」から出発して、より難易度の高い偽造に対抗するための識別基準を構築しました。

本章では、生理信号、3D 幾何学、音響知覚に跨る代表的な 5 篇の論文を紹介し、多モーダル FAS 技術の潜力と将来性を示します。

26. [**[16.12] Generalized face anti-spoofing by detecting pulse from face videos**](https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICPR-2016/media/files/1223.pdf)
    初期の FAS シナリオにおいて、深度センサーや赤外線センサーがなくても、顔の心拍信号だけで偽顔を識別できる方法が示され、rPPG の潜在能力を強調しています。

    :::info
    **論文ノート**：[**[16.12] rPPG: 生命の光斑**](https://docsaid.org/ja/papers/face-antispoofing/rppg)
    :::

27. [**[18.09] Remote Photoplethysmography Correspondence Feature for 3D Mask Face Presentation Attack Detection**](https://dl.acm.org/doi/10.1007/978-3-030-01270-0_34)
    初めて CFrPPG（対応型 rPPG）特徴を提案し、低光量やカメラの揺れなどの条件下でも心拍軌跡を正確に抽出。3D マスク攻撃に対して優れたパフォーマンスを発揮。

    :::info
    **論文ノート**：[**[18.09] CFrPPG: 心拍の残響**](https://docsaid.org/ja/papers/face-antispoofing/cfrppg)
    :::

28. [**[19.05] Multi-Modal Face Authentication Using Deep Visual and Acoustic Features**](https://ieeexplore.ieee.org/document/8761776)
    スマートフォン内蔵のスピーカーとマイクを使用して超音波を発射し、顔面エコーを解析。CNN で抽出した画像特徴と組み合わせ、追加のハードウェアなしで二重モーダルセキュリティ認証システムを構築。

    :::info
    **論文ノート**：[**[19.05] VA-FAS: 音波の中の顔**](https://docsaid.org/ja/papers/face-antispoofing/vafas)
    :::

29. [**[22.08] Beyond the Pixel World: A Novel Acoustic-Based Face Anti-Spoofing System for Smartphones**](https://ieeexplore.ieee.org/document/9868051)
    Echo-Spoof という音響 FAS データセットを構築し、Echo-FAS アーキテクチャを設計。音波を使用して 3D 幾何学と材料情報を再構築し、カメラに依存せず、モバイルデバイスにおける低コスト・高耐性のアプリケーション事例。

    :::info
    **論文ノート**：[**[22.08] Echo-FAS: 偽造のエコー**](https://docsaid.org/ja/papers/face-antispoofing/echo-fas)
    :::

30. [**[24.03] AFace: Range-Flexible Anti-Spoofing Face Authentication via Smartphone Acoustic Sensing**](https://dl.acm.org/doi/10.1145/3643510)
    Echo-FAS のアイデアを拡張し、iso-depth モデルと距離適応アルゴリズムを追加。3D プリントマスクに対抗し、ユーザーの距離に応じて自動調整。音波による活体認証の実用化への重要な設計。

    :::info
    **論文ノート**：[**[24.03] AFace: 波動の邊界**](https://docsaid.org/ja/papers/face-antispoofing/aface)
    :::

---

これらの 5 本の論文は、非視覚モーダルが FAS 分野における重要な始まりを築いたものであり、従来のカメラの制限を避けるために深く掘り下げるべき方向性です。

## 第七章：偽りの軌跡を解体する

> **偽装の構造と意味を深くモデル化し、モデルの識別力を高める**

FAS（顔認証のなりすまし防止）モデルが、解釈可能性と汎化能力という二つの課題に直面する中で、研究者たちは「spoof trace（偽装痕跡）」という概念に注目し始めた。これは、偽顔が映像に残す微細なパターン、例えば色のずれや輪郭の異常、周波数の異変などを指す。

本章の 5 本の論文は、表現の分離（disentanglement）の観点からアプローチし、偽装特徴を顔の本来の情報から切り離すことで、偽装サンプルの再構築・解析・さらには合成までを可能にし、モデルが「偽装を見抜く」ことを学習することを目指している。

31. [**[20.03] Searching Central Difference Convolutional Networks for Face Anti-Spoofing**](https://arxiv.org/abs/2003.04092)
    中心差分（CDC）手法を提案。人工的に「偽装は局所的な勾配に差異を残すべきである」という仮説を定義し、実際の顔と潜在的な偽装の勾配信号を分離する。さらに多尺度注意モジュールを組み合わせることで、高効率なデプロイとクロスデータセットでの一般化能力を実現した FAS 手法であり、非常に多く引用されている。

    :::info
    **論文ノート**：[**[20.03] CDCN: 真と偽の入り混じる境界**](https://docsaid.org/ja/papers/face-antispoofing/cdcn)
    :::

32. [**[20.07] On Disentangling Spoof Trace for Generic Face Anti-Spoofing**](https://arxiv.org/abs/2007.09273)
    多尺度で偽装痕跡を分離するモデルを提案。偽装信号を多層のパターンの組み合わせと捉え、敵対的学習を通じて本物の顔と偽装マスクを再構築。新たな攻撃サンプルの合成にも応用できる、偽装認識表現学習の代表的研究。

    :::info
    **論文ノート**：[**[20.07] STDN: 偽装の痕跡**](https://docsaid.org/ja/papers/face-antispoofing/stdn)
    :::

33. [**[20.08] Face Anti-Spoofing via Disentangled Representation Learning**](https://arxiv.org/abs/2008.08250)
    顔の特徴を「生体（liveness）」と「個人識別（identity）」の 2 つのサブ空間に分解。CNN アーキテクチャで低次・高次信号を分離し、より転移可能な生体分類器を構築。異なる攻撃タイプに対する安定性を向上。

    :::info
    **論文ノート**：[**[20.08] Disentangle-FAS: 魂の結び目を解く**](https://docsaid.org/ja/papers/face-antispoofing/disentangle-fas)
    :::

34. [**[21.10] Disentangled representation with dual-stage feature learning for face anti-spoofing**](https://arxiv.org/abs/2110.09157)
    二段階の分離学習機構を用い、顔画像を生体に関連する部分と無関係な部分の 2 つのサブ空間に分離。未知の攻撃タイプに対する認識能力を効果的に向上させ、汎化性能強化の重要な設計。

    :::info
    **論文ノート**：[**[21.10] DualStage: 複解耦の術**](https://docsaid.org/ja/papers/face-antispoofing/dualstage)
    :::

35. [**[21.12] Dual spoof disentanglement generation for face anti-spoofing with depth uncertainty learning**](https://arxiv.org/abs/2112.00568)
    DSDG 生成フレームワークを提案。VAE を用いて個人識別と攻撃テクスチャの因子化潜在表現を学習し、多様な偽装画像を大規模に合成可能。深度不確実性モジュールを導入し深度監督の安定化を図る、「生成対抗偽装」の代表例。

    :::info
    **論文ノート**：[**[21.12] DSDG: 偽りの再構成の前夜**](https://docsaid.org/ja/papers/face-antispoofing/dsdg)
    :::

---

本章は重要な転換点を示している。すなわち「生体検出」から「偽装解析」へ、そして「攻撃のシミュレーション」へと、FAS 研究は徐々に「生成可能・解釈可能・制御可能」という次の段階へと進化している。これらの手法はモデルの精度向上だけでなく、将来の攻防の進化の道筋を示唆する可能性を秘めている。

## 第八章：未来の混沌

> **CLIP から人間の知覚へ、FAS の次の境界**

単一モーダル、単一攻撃タイプだけでは実際のニーズを満たすのが難しくなったとき、FAS はさらに高次の挑戦に直面しています：**物理的+デジタルな二重攻撃、セマンティクス駆動の識別、さまざまな環境でのゼロショット一般化**。

これらの 5 本の代表作は、FAS の未来に向けた 3 つの主要な発展軸：**融合識別、言語モデル、そして人間の感知**を示しています。

36. [**[23.09] FLIP: Cross-domain Face Anti-Spoofing with Language Guidance**](https://arxiv.org/abs/2309.16649)
    CLIP モデルを FAS タスクに応用し、自然言語による記述で視覚的特徴空間を導く。クロスドメインでの一般化能力を向上させ、セマンティックアライメントと多モーダル対比学習戦略を提案。言語駆動でのゼロショット FAS を実現。

37. [**[24.04] Joint Physical-Digital Facial Attack Detection via Simulating Spoofing Clues**](https://arxiv.org/abs/2404.08450)
    SPSC と SDSC データ拡張戦略を提案し、物理的およびデジタル攻撃の手がかりをシミュレート。単一のモデルで両方の攻撃タイプを識別できるようにし、CVPR2024 コンペで優勝。融合型モデルの新たな基準を打ち立てました。

38. [**[24.04] Unified Physical-Digital Attack Detection Challenge**](https://arxiv.org/abs/2404.06211)
    初の統一攻撃識別挑戦コンペを立ち上げ、2.8 万件の複合型攻撃データセット UniAttackData を公開。各チームのモデルアーキテクチャを分析し、Unified Attack Detection への道を開くカタリストとなりました。

39. [**[24.08] La-SoftMoE CLIP for Unified Physical-Digital Face Attack Detection**](https://arxiv.org/abs/2408.12793)
    CLIP と Mixture of Experts アーキテクチャを組み合わせ、soft-adaptive メカニズムを導入してサブモデルを動的に割り当て、複雑な意思決定境界に対応。物理的およびデジタル攻撃の融合処理に効率的なパラメータ選択を提供。

40. [**[25.01] Interpretable Face Anti-Spoofing: Enhancing Generalization with Multimodal Large Language Models**](https://arxiv.org/abs/2501.01720)
    マルチモーダル大型言語モデルを統合した新しいフレームワーク I-FAS を提案。顔の活体認証タスクを解釈可能な視覚的質問応答問題に変換し、意味注釈、非対称言語損失、グローバル認識コネクタという三つの重要な設計を通じて、モデルのドメイン間一般化能力と推論性能を大幅に向上させている。

---

この章は FAS 分野の未来のトレンドを象徴しています：**偽顔の識別 → 攻撃タイプの推測 → セマンティクスの理解 → 多モーダル言語論理推論との統合**。研究は「視覚理解」から「セマンティック認知」へと進化しており、攻撃も単一のモデルから複雑な混合型に進化しています。

## 結語

現実世界で最も不足しているものは悪意であり、顔認識のニーズがある限り、防偽のニーズは止まることはありません。

最初のテクスチャ解析、光影モデリングから畳み込みネットワークの登場、さらに ViT、CLIP、音波、そして人間の知覚の導入に至るまで、FAS 技術はその境界を拡大し続けています。これらの論文は、単なる古典やトレンドの整理にとどまらず、数十年にわたる技術進化の地図であり、過去、現在、未来を繋げるものです。

この地図の中で私たちは以下のことを見ています：

- **単一モーダルから多モーダルへ**：画面を見るだけでなく、深度、音、脈動、材質も感知する。
- **分類から解耦へ**：真偽を判定するだけでなく、各偽装の構成方法を理解しようとしている。
- **識別から推論へ**：活体を識別するだけでなく、セマンティクス、材料、そして言語記述の背後にある真実を理解しようとしている。
- **防御から生成へ**：単に受動的な防御にとどまらず、積極的に模擬、再構築、干渉するようになった。

この分野に足を踏み入れようとしているあなたにとって、この技術ガイドは「万能の解決策」を提供するものではありませんが、出発点を見つける手助けになるでしょう。
