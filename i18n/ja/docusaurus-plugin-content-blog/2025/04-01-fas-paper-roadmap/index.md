---
slug: fas-paper-roadmap
title: Face Anti-Spoofing 技術地図
authors: Z. Yuan
image: /ja/img/2025/0401.jpg
tags: [face-anti-spoofing, liveness-detection]
description: 伝統から未来への40本の論文ガイド。
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

2. [**[12.09] On the Effectiveness of Local Binary Patterns in Face Anti-Spoofing**](https://ieeexplore.ieee.org/document/6313548)
   LBP とその変種を用いて、平面写真とスクリーン再生攻撃を識別し、REPLAY-ATTACK データセットを構築。最も初期の公開データセットおよび古典的なベースラインの組み合わせの一つです。

3. [**[14.05] Spoofing Face Recognition with 3D Masks**](https://ieeexplore.ieee.org/document/6810829)
   3D マスクが異なる顔認識システム（2D/2.5D/3D）に対する攻撃効果を系統的に分析し、伝統的な平面顔に対する仮定が 3D 印刷技術では成り立たないことを指摘。

4. [**[19.09] Biometric Face Presentation Attack Detection with Multi-Channel Convolutional Neural Network**](https://arxiv.org/abs/1909.08848)
   RGB、深度、赤外線、熱感知信号を組み合わせた多チャネル CNN アーキテクチャを提案し、WMCA データセットを発表。高次の偽顔（シリコンマスクなど）の検出能力を向上。

5. [**[22.10] Deep Learning for Face Anti-Spoofing: A Survey**](https://ieeexplore.ieee.org/abstract/document/9925105)
   FAS 分野で初めての深層学習に基づいた系統的なレビュー論文。ピクセル単位の監視、多モーダルセンサー、ドメイン一般化など新しいトレンドを取り上げ、知識の全体像を構築。

---

これらの手法は単純であるものの、平面顔（写真やスクリーン再生）の認識基盤を築くものであり、後の深層学習技術導入に向けた概念的枠組みを作り上げました。

## 第二章：現実世界の舞台

> **FAS 技術が実験室から現実のシーンに進出するマイルストーン**

データセットとベンチマークは、ある分野が安定的に成長できるかどうかを決定します。

FAS 技術は、単一のシーンから複数のデバイス、複数の光源、複数の攻撃手法に対応するようになり、これらの代表的な公開データセットによって推進されてきました。

6. [**[17.06] OULU-NPU: A Mobile Face Presentation Attack Database with Real-World Variations**](https://ieeexplore.ieee.org/document/7961798)
   モバイルシーン向けに設計された FAS データセットで、デバイス、環境光、攻撃手法などのさまざまな変数を含み、4 つのテストプロトコルを設計。これにより「一般化能力」の評価が可能になったマイルストーン。

7. [**[20.03] CASIA-SURF CeFA: A Benchmark for Multi-modal Cross-ethnicity Face Anti-Spoofing**](https://arxiv.org/abs/2003.05136)
   世界初の「民族タグ付け」のある大型多モーダル FAS データセットで、RGB、深度、IR および複数の攻撃タイプを含み、特に民族偏向とモーダル融合戦略の研究に役立つ。

8. [**[20.07] CelebASpoof: Large-scale Face Anti-Spoofing Dataset with Rich Annotations**](https://arxiv.org/abs/2007.12342)
   現在最大規模の FAS データセットで、62 万枚以上の画像を含み、10 種類の spoof タグと元の CelebA の 40 の属性が含まれており、多タスクおよび spoof トレース学習に適しています。

9. [**[22.01] A Personalized Benchmark for Face Anti-Spoofing**](https://openaccess.thecvf.com/content/WACV2022W/MAP-A/html/Belli_A_Personalized_Benchmark_for_Face_Anti-Spoofing_WACVW_2022_paper.html)
   ユーザー登録時の活体画像を識別プロセスに組み込む提案。CelebA-Spoof-Enroll および SiW-Enroll という 2 つの新しいテスト設定を提案し、個人化 FAS システムの可能性を探る。

10. [**[24.02] SHIELD: An Evaluation Benchmark for Face Spoofing and Forgery Detection with Multimodal Large Language Models**](https://arxiv.org/abs/2402.04178)
    LLM と多モーダル入力を組み合わせ、QA タスク形式で MLLM の spoof/forgery 検出における推論能力を評価。攻撃を「言語モデリングで理解する」という新しい領域を開拓。

## 第三章：跨領域の修羅場

> **単一データ学習から多シーン展開の核心技術**

Face Anti-Spoofing（FAS）の最も難しい問題の一つは、一般化能力です——モデルが訓練データだけでなく、新しいデバイス、新しい環境、新しい攻撃にも対応できるようにする方法。

11. [**[20.04] Single-Side Domain Generalization for Face Anti-Spoofing**](https://arxiv.org/abs/2004.14043)
    単一の対抗学習戦略を提案し、真顔のみでドメイン間調整を行い、偽顔の特徴を異なるドメインで自然に分散させることで、誤った情報の過度な圧縮を避ける。これは DG 設計における非常に示唆に富んだ方向性です。

12. [**[21.05] Generalizable Representation Learning for Mixture Domain Face Anti-Spoofing**](https://arxiv.org/abs/2105.02453)
    ドメインラベルを既知とせず、インスタンス正規化と MMD を使用して、無監督のクラスタリングと調整を実現。人工的なクラスタリングに依存しない一般化訓練フローを実現。

13. [**[23.03] Rethinking Domain Generalization for Face Anti-Spoofing: Separability and Alignment**](https://arxiv.org/abs/2303.13662)
    SA-FAS フレームワークを提案し、異なるドメインで特徴の分離性を保ちながら、live→spoof の変化軌跡が各ドメインで一貫するように強調。これは IRM 理論を FAS に深く適用したものです。

14. [**[24.02] Suppress and Rebalance: Towards Generalized Multi-Modal Face Anti-Spoofing**](https://arxiv.org/abs/2402.19298)
    多モーダル DG 問題を深く分析し、U-Adapter を使用して不安定なモーダルの干渉を抑制し、ReGrad で各モーダルの収束速度を動的に調整することで、モーダル不均衡と信頼性の問題に対する完全な解決策を提供。

15. [**[24.04] VL-FAS: Domain Generalization via Vision-Language Model for Face Anti-Spoofing**](https://ieeexplore.ieee.org/document/10448156)
    初めて Vision-Language メカニズムを導入し、意味的に顔の領域に注意を集中させることで、SLVT による意味的層の一般化を行い、ViT のクロスドメイン安定性を大幅に向上。

---

これらの 5 本の論文は、現在の Domain Generalization（DG）テーマの技術的軸を構成しています。単一の対抗、ラベルなしクラスタリング、分離性分析から、言語を統合した監視方法に至るまで、クロスドメインの課題に対する完全な戦略を描き出しています。

## 第四章：新世界の勃興

> **CNN から ViT へ、FAS モデルのアーキテクチャ革新の道**

Vision Transformer（ViT）の登場により、画像タスクは局所的な畳み込みから全体的なモデリング時代へと進化しました。Face Anti-Spoofing も例外ではありません。

16. [**[23.02] Rethinking Vision Transformer and Masked Autoencoder in Multimodal Face Anti-Spoofing**](https://arxiv.org/abs/2302.05744)
    ViT が多モーダル FAS における主要な問題を全面的に再考。入力設計、事前学習戦略、パラメータ微調整フローを含む、AMA アダプターと M2A2E 事前学習アーキテクチャを提案し、クロスモーダルかつラベルなしの自己監督プロセスを構築。

17. [**[23.04] Ma-ViT: Modality-Agnostic Vision Transformers for Face Anti-Spoofing**](https://arxiv.org/abs/2304.07549)
    単一分岐の早期融合アーキテクチャを採用し、Modal-Disentangle Attention と Cross-Modal Attention を通じて、モーダルに依存しない識別能力を実現。記憶効率と柔軟な展開を両立させた、ViT の実用性における重要な一歩。

18. [**[23.05] FM-ViT: Flexible Modal Vision Transformers for Face Anti-Spoofing**](https://arxiv.org/abs/2305.03277)
    モーダル欠損と高精度攻撃の問題を解決するために、クロスモーダルアテンション設計（MMA + MFA）を提案。各モーダルの特徴を保ちつつ、偽顔のパッチへの焦点を強化する、展開の柔軟性を考慮した設計のテンプレート。

19. [**[23.09] Sadapter: Generalizing Vision Transformer for Face Anti-Spoofing with Statistical Tokens**](https://arxiv.org/abs/2309.04038)
    Efficient Parameter Transfer Learning アーキテクチャを利用して、ViT に統計的アダプターを挿入し、主ネットワークのパラメータを固定。Token Style Regularization でスタイル差を抑制し、クロスドメイン FAS に特化した軽量ソリューション。

20. [**[23.10] LDCFormer: Incorporating Learnable Descriptive Convolution to Vision Transformer for Face Anti-Spoofing**](https://ieeexplore.ieee.org/document/10222330)
    学習可能な記述的畳み込み（LDC）を ViT に統合し、局所的な詳細表現能力を強化。複数のベンチマークで SOTA（最先端技術）を達成するために、最適化されたバージョン LDCformerD を提案。

---

この段階の 5 本の論文は、Transformer アーキテクチャが多モーダル入力、モーダル欠損、クロスドメインスタイル、局所パッチ表現などの重要な課題をどのように処理しているかを示しています。これは FAS モデル設計ロジックの全面的な転換を表しています。

## 第五章：風格の戦い

> **異なる世界からの spoof、風格に敏感でないモデルをどのように作成するか？**

FAS モデルの一般化能力は、ドメインシフトの挑戦だけでなく、異なるスタイル（style）間の情報不対称の干渉を受けます。

この章では、スタイルの解耦、対抗学習、テスト時適応（test-time adaptation）およびインスタンスアウェア設計に焦点を当てています。これらの方法は、モデルが未知のスタイルやサンプル分布でも安定した識別性能を維持できるようにすることを試みています。

21. [**[21.07] Unified Unsupervised and Semi-Supervised Domain Adaptation Network for Cross-Scenario Face Anti-Spoofing**](https://www.sciencedirect.com/science/article/abs/pii/S0031320321000753)
    USDAN フレームワークを提案し、無監督および半監督設定を同時にサポート。境界と条件調整モジュールを通じて、対抗訓練を行い、異なるタスク設定に対応できる一般化表現を学習。

22. [**[22.03] Domain Generalization via Shuffled Style Assembly for Face Anti-Spoofing**](https://arxiv.org/abs/2203.05340)
    内容とスタイルを分離する戦略を採用し、スタイル空間を再構成してスタイルシフトをシミュレート。対比学習を通じて、ライブ顔に関連するスタイルを強調することは、スタイル認識を伴う DG 設計の重要な突破口。

23. [**[23.03] Adversarial Learning Domain-Invariant Conditional Features for Robust Face Anti-Spoofing**](https://link.springer.com/article/10.1007/s11263-023-01778-x)
    境界分布の調整だけでなく、条件調整の対抗構造を導入し、クラス単位で区別可能なクロスドメイン共有表現を学習。誤った調整問題を効果的に解決。

24. [**[23.03] Style Selective Normalization with Meta Learning for Test-Time Adaptive Face Anti-Spoofing**](https://www.sciencedirect.com/science/article/abs/pii/S0957417422021248)
    統計情報を利用して入力画像のスタイルを推定し、テスト時に適応的に正規化パラメータを選択。メタ学習を組み合わせて未知のドメインへの転送プロセスを事前にシミュレート。

25. [**[23.04] Instance-Aware Domain Generalization for Face Anti-Spoofing**](https://arxiv.org/abs/2304.05640)
    粗いドメインラベルを放棄し、インスタンスレベルのスタイル調整戦略を採用。非対称ホワイトニング、スタイル強化、動的カーネル設計を通じて、スタイルに敏感でない識別特徴を抽出。

---

これらの 5 本の論文は、さまざまな角度から「スタイル一般化」というテーマに挑戦しており、特にインスタンスベースおよびテスト時適応の試みでは、実際のアプリケーションシナリオに近づいています。

## 第六章：多モーダルの召喚術

> **画像だけではなく、音声や生理信号も登場**

従来の RGB モデルが高精度攻撃やクロスドメインの課題に直面した時、FAS コミュニティは視覚以外の信号、例えば**rPPG、生理信号、音波エコー**などの補助情報を探索し、「人間の信号」から出発して、より難易度の高い偽造に対抗するための識別基準を構築しました。

本章では、生理信号、3D 幾何学、音響知覚に跨る代表的な 5 篇の論文を紹介し、多モーダル FAS 技術の潜力と将来性を示します。

26. [**[18.09] Remote Photoplethysmography Correspondence Feature for 3D Mask Face Presentation Attack Detection**](https://dl.acm.org/doi/10.1007/978-3-030-01270-0_34)
    初めて CFrPPG（対応型 rPPG）特徴を提案し、低光量やカメラの揺れなどの条件下でも心拍軌跡を正確に抽出。3D マスク攻撃に対して優れたパフォーマンスを発揮。

27. [**[19.05] Multi-Modal Face Authentication Using Deep Visual and Acoustic Features**](https://ieeexplore.ieee.org/document/8761776)
    スマートフォン内蔵のスピーカーとマイクを使用して超音波を発射し、顔面エコーを解析。CNN で抽出した画像特徴と組み合わせ、追加のハードウェアなしで二重モーダルセキュリティ認証システムを構築。

28. [**[21.04] Contrastive Context-Aware Learning for 3D High-Fidelity Mask Face Presentation Attack Detection**](https://arxiv.org/abs/2104.06148)
    高精度 3D マスクの課題を解決するために、HiFiMask という大規模データセットを構築し、CCL 対比学習法を提案。文脈情報（人物、素材、光）を利用して攻撃識別能力を向上。

29. [**[22.08] Beyond the Pixel World: A Novel Acoustic-Based Face Anti-Spoofing System for Smartphones**](https://ieeexplore.ieee.org/document/9868051)
    Echo-Spoof という音響 FAS データセットを構築し、Echo-FAS アーキテクチャを設計。音波を使用して 3D 幾何学と材料情報を再構築し、カメラに依存せず、モバイルデバイスにおける低コスト・高耐性のアプリケーション事例。

30. [**[24.03] AFace: Range-Flexible Anti-Spoofing Face Authentication via Smartphone Acoustic Sensing**](https://dl.acm.org/doi/10.1145/3643510)
    Echo-FAS のアイデアを拡張し、iso-depth モデルと距離適応アルゴリズムを追加。3D プリントマスクに対抗し、ユーザーの距離に応じて自動調整。音波による活体認証の実用化への重要な設計。

---

これらの 5 本の論文は、非視覚モーダルが FAS 分野における重要な始まりを築いたものであり、従来のカメラの制限を避けるために深く掘り下げるべき方向性です。

## 第七章：偽りの軌跡を解体する

> **spoof の構造とセマンティクスを深くモデル化し、モデルの識別能力を向上させる**

FAS モデルが解釈性と一般化能力の両方の挑戦に向かって進む中で、研究者たちは「spoof trace」という概念に注目し始めました。これは、偽顔が画像に残す微細なパターン、例えば色の偏差、エッジの輪郭、周波数の異常などを指します。

この章の 5 本の論文はすべて**特徴の解耦**（disentanglement）の観点からアプローチし、spoof 特徴を顔の内容から分離し、偽装を再構築、分析、さらには合成することで、モデルが「偽装を見抜く」方法を学べるようにしています。

31. [**[20.07] On Disentangling Spoof Trace for Generic Face Anti-Spoofing**](https://arxiv.org/abs/2007.09273)
    複数スケールの spoof trace 分離モデルを提案し、偽装信号を多層パターンの組み合わせとして捉え、対抗学習を通じて実際の顔と spoof マスクを再構築。新たな攻撃サンプルを合成するために応用可能で、spoof-aware な特徴学習の代表的な作品。

32. [**[20.08] Face Anti-Spoofing via Disentangled Representation Learning**](https://arxiv.org/abs/2008.08250)
    顔の特徴を liveness と identity の 2 つのサブスペースに分解し、CNN アーキテクチャを使用して低次および高次の信号を分離。より移行性のある活体分類器を構築し、さまざまな攻撃タイプに対する安定性を向上。

33. [**[22.03] Spoof Trace Disentanglement for Generic Face Anti-Spoofing**](https://ieeexplore.ieee.org/document/9779478)
    spoof trace を加算可能かつ修復可能なパターンとしてモデル化し、2 段階の解耦フレームワークを提案。周波数領域情報を統合することで、低次の spoof 検出能力を強化。spoof データ増強にも使用可能で、long-tail 攻撃の一般化を向上。

34. [**[22.07] Learning to Augment Face Presentation Attack Dataset via Disentangled Feature Learning from Limited Spoof Data**](https://ieeexplore.ieee.org/document/9859657)
    少量の spoof サンプルに対して解耦式の remix 戦略を提案。分離後の liveness と identity 特徴空間で生成し、対比学習を使用して識別性を維持。小規模サンプル環境での識別性能を大幅に向上。

35. [**[22.12] Learning Polysemantic Spoof Trace: A Multi-Modal Disentanglement Network for Face Anti-Spoofing**](https://arxiv.org/abs/2212.03943)
    spoof trace 解耦アーキテクチャを多モーダルに拡張し、RGB/Depth 二重経路ネットワークを設計して補完的な spoof 手がかりをキャプチャ。クロスモダリティ融合を通じて両者のセマンティクスを組み合わせ、汎用 FAS モデルの先駆けとなる提案。

---

この章では重要な転換点が示されています：活体の識別 → 偽装の分析 → 攻撃のシミュレーション。Face Anti-Spoofing の研究は「生成可能、解釈可能、操作可能」な次の段階へと進んでおり、これらの方法はモデルの精度向上に貢献するだけでなく、将来の攻防の進化の道を切り開くかもしれません。

## 第八章：未来の混沌

> **CLIP から人間の知覚へ、FAS の次の境界**

単一モーダル、単一攻撃タイプだけでは実際のニーズを満たすのが難しくなったとき、FAS はさらに高次の挑戦に直面しています：**物理的+デジタルな二重攻撃、セマンティクス駆動の識別、さまざまな環境でのゼロショット一般化**。

これらの 5 本の代表作は、FAS の未来に向けた 3 つの主要な発展軸：**融合識別、言語モデル、そして人間の感知**を示しています。

36. [**[20.07] Face Anti-Spoofing with Human Material Perception**](https://arxiv.org/abs/2007.02157)
    材質知覚を FAS モデル設計に取り入れ、BCN アーキテクチャを用いて人間の知覚レベル（マクロ/ミクロ）で材質差を判定。皮膚、紙、シリコンなどの材質差を軸に、モデルのセマンティックな解釈性と材質間識別能力を向上。

37. [**[23.09] FLIP: Cross-domain Face Anti-Spoofing with Language Guidance**](https://arxiv.org/abs/2309.16649)
    CLIP モデルを FAS タスクに応用し、自然言語による記述で視覚的特徴空間を導く。クロスドメインでの一般化能力を向上させ、セマンティックアライメントと多モーダル対比学習戦略を提案。言語駆動でのゼロショット FAS を実現。

38. [**[24.04] Joint Physical-Digital Facial Attack Detection via Simulating Spoofing Clues**](https://arxiv.org/abs/2404.08450)
    SPSC と SDSC データ拡張戦略を提案し、物理的およびデジタル攻撃の手がかりをシミュレート。単一のモデルで両方の攻撃タイプを識別できるようにし、CVPR2024 コンペで優勝。融合型モデルの新たな基準を打ち立てました。

39. [**[24.04] Unified Physical-Digital Attack Detection Challenge**](https://arxiv.org/abs/2404.06211)
    初の統一攻撃識別挑戦コンペを立ち上げ、2.8 万件の複合型攻撃データセット UniAttackData を公開。各チームのモデルアーキテクチャを分析し、Unified Attack Detection への道を開くカタリストとなりました。

40. [**[24.08] La-SoftMoE CLIP for Unified Physical-Digital Face Attack Detection**](https://arxiv.org/abs/2408.12793)
    CLIP と Mixture of Experts アーキテクチャを組み合わせ、soft-adaptive メカニズムを導入してサブモデルを動的に割り当て、複雑な意思決定境界に対応。物理的およびデジタル攻撃の融合処理に効率的なパラメータ選択を提供。

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
