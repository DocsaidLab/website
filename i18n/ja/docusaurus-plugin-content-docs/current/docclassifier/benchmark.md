---
sidebar_position: 6
---

# モデルの評価

:::warning
本プロジェクトのテストデータセットは、民間機関から提供されたものです。プライバシー保護のため、このデータセットの評価結果のみを提供しています。
:::

このデータセットには約 25,000 枚の「テキストロケーション切り抜き」が施された「去識別化」画像が含まれており、7 種類の異なるカテゴリがあり、その数は極端に不均衡です。データには、光の変化、ぼやけ、反射、角点の位置誤差などによる切り抜きの歪みが多く含まれています。

私たちはこのデータセットの「誤ったカテゴリラベル」のみをクリーンアップし、その後すべてのデータを使ってモデルのパフォーマンスを評価しています。

## 評価プロトコル

### AUROC

AUROC (Receiver Operating Characteristic Curve 下の面積) は、分類モデルのパフォーマンスを評価するための統計指標で、特に二項分類問題を扱う際に使用されます。AUROC の値は 0 から 1 の範囲で、AUROC の値が高いほど、モデルは 2 つのカテゴリをうまく区別する能力があることを示します。

- **ROC 曲線**

  - **定義**：ROC 曲線は、分類モデルがすべての可能な分類しきい値に対してどのように動作するかを示すグラフィカルな評価ツールです。これにより、真陽性率 (TPR) と偽陽性率 (FPR) を異なるしきい値でプロットして表示します。
  - **真陽性率（TPR）**：感度とも呼ばれ、計算式は TPR = TP / (TP + FN) です。ここで TP は真陽性の数、FN は偽陰性の数です。
  - **偽陽性率（FPR）**：計算式は FPR = FP / (FP + TN) です。ここで FP は偽陽性の数、TN は真陰性の数です。

- **AUROC の計算**

  - AUROC は ROC 曲線の下の面積です。これにより、モデルがすべての分類しきい値に対するパフォーマンスを総括的に評価する指標が得られます。
  - **分析方法**：
    - **AUROC = 1**：完璧な分類器で、2 つのカテゴリを完全に区別できます。
    - **0.5 < AUROC < 1**：モデルはある程度の区別能力を持っており、AUROC の値が 1 に近いほどモデルの性能が良いことを示します。
    - **AUROC = 0.5**：区別能力がなく、ランダムな予測と同等です。
    - **AUROC < 0.5**：ランダム予測よりも悪いですが、モデルが予測を反転解釈すれば、良い性能が得られる可能性があります。

### TPR@FPR しきい値表

TPR@FPR しきい値表は、顔認識分野で広く使用されている重要な評価ツールで、モデルのパフォーマンスを異なるしきい値設定で測定するために使用されます。この表は ROC 曲線から派生しており、モデルのパフォーマンスを直感的かつ正確に評価する方法を提供します。

例えば、FPR（偽陽性率）が 0.01 のときに TPR（真陽性率）が少なくとも 0.9 になるパフォーマンスを目標とする場合、TPR-FPR しきい値表を使って対応するしきい値を特定できます。このしきい値が、モデル推論プロセスをガイドします。

本プロジェクトの実装でも、同様の評価方法を採用しています。私たちは FPR が 0.0001 のときの TPR のパフォーマンスを基準として選びました。この基準により、特定の条件下でモデルのパフォーマンスをより正確に理解することができます。

### ゼロショットテスト

私たちはゼロショットテスト戦略を採用し、テストデータに含まれるすべてのカテゴリやパターンが訓練データには現れないことを保証します。これは、モデルの訓練段階でテストセットのサンプルやカテゴリを一切学習していないことを意味します。このアプローチは、モデルが完全に未知のデータに対してどのように一般化し、識別する能力を評価し、検証することを目的としています。

このテスト方法は、ゼロショット学習（Zero-shot Learning）モデルの評価に特に適しています。ゼロショット学習の核心的な課題は、訓練中に一度も見たことのないカテゴリを扱うことです。ゼロショット学習の文脈では、モデルは通常、他の補助的な情報（カテゴリのテキスト記述、属性ラベル、カテゴリ間の意味的関係など）を利用して、新しいカテゴリを理解する必要があります。したがって、ゼロショットテストでは、モデルは訓練されたカテゴリから学んだ知識と、カテゴリ間の潜在的な関係を活用して、テストセットの新しいサンプルを識別します。

## アブレーション実験

- **グローバル設定**

  - クラス数: 394,080
  - エポック数: 20
  - 1 エポックあたりのデータ数: 2,560,000
  - バッチサイズ: 512
  - 最適化アルゴリズム: AdamW
  - 設定：
    - flatten: Flatten -> Linear (デフォルト)
    - gap: GlobalAveragePooling2d -> Linear
    - squeeze: Conv2d -> Flatten -> Linear

- **総合比較**

  | 名前                                   | TPR@FPR=1e-4 |    ROC     | FLOPs (G) | サイズ (MB) |
  | -------------------------------------- | :----------: | :--------: | :-------: | :---------: |
  | lcnet050-f256-r128-ln-arc              |    0.754     |   0.9951   |   0.053   |    5.54     |
  | lcnet050-f256-r128-ln-softmax          |    0.663     |   0.9907   |   0.053   |    5.54     |
  | lcnet050-f256-r128-ln-cos              |  **0.784**   | **0.9968** |   0.053   |    5.54     |
  | lcnet050-f256-r128-ln-cos-from-scratch |    0.141     |   0.9273   |   0.053   |    5.54     |
  | lcnet050-f256-r128-ln-cos-squeeze      |    0.772     |   0.9958   |   0.052   |  **2.46**   |
  | lcnet050-f256-r128-bn-cos              |    0.721     |   0.992    |   0.053   |    5.54     |
  | lcnet050-f128-r96-ln-cos               |    0.713     |   0.9944   |   0.029   |    2.33     |
  | lcnet050-f256-r128-ln-cos-gap          |    0.480     |   0.9762   |   0.053   |    2.67     |
  | efficientnet_b0-f256-r128-ln-cos       |    0.682     |   0.9931   |   0.242   |    19.89    |

- **目標カテゴリ数の比較**

  | 名前                      | クラス数 | TPR@FPR=1e-4 |    ROC     |
  | ------------------------- | -------: | :----------: | :--------: |
  | lcnet050-f256-r128-ln-arc |   16,256 |    0.615     |   0.9867   |
  | lcnet050-f256-r128-ln-arc |  130,048 |    0.666     |   0.9919   |
  | lcnet050-f256-r128-ln-arc |  390,144 |  **0.754**   | **0.9951** |

  - クラス数が多いほど、モデルのパフォーマンスが向上します。

- **MarginLoss 比較**

  | 名前                          | TPR@FPR=1e-4 |    ROC     |
  | ----------------------------- | :----------: | :--------: |
  | lcnet050-f256-r128-ln-softmax |    0.663     |   0.9907   |
  | lcnet050-f256-r128-ln-arc     |    0.754     |   0.9951   |
  | lcnet050-f256-r128-ln-cos     |  **0.784**   | **0.9968** |

  - CosFace または ArcFace を単独で使用する場合、ArcFace の方が効果的です。
  - PartialFC を組み合わせると、CosFace の方が効果的です。

- **BatchNorm vs LayerNorm**

  | 名前                      | TPR@FPR=1e-4 |    ROC     |
  | ------------------------- | :----------: | :--------: |
  | lcnet050-f256-r128-bn-cos |    0.721     |   0.9921   |
  | lcnet050-f256-r128-ln-cos |  **0.784**   | **0.9968** |

  - LayerNorm を使用する方が BatchNorm よりも効果的です。

- **Pretrain vs From-Scratch**

  | 名前                                   | TPR@FPR=1e-4 |    ROC     |
  | -------------------------------------- | :----------: | :--------: |
  | lcnet050-f256-r128-ln-cos-from-scratch |    0.141     |   0.9273   |
  | lcnet050-f256-r128-ln-cos              |  **0.784**   | **0.9968** |

  - Pretrain を使用することが必須であり、大幅に時間を節約できます。

- **モデルサイズの削減方法**

  | 名前                              | TPR@FPR=1e-4 |    ROC     | サイズ (MB) | FLOPs (G) |
  | --------------------------------- | :----------: | :--------: | :---------: | :-------: |
  | lcnet050-f256-r128-ln-cos         |  **0.784**   | **0.9968** |    5.54     |   0.053   |
  | lcnet050-f256-r128-ln-cos-squeeze |    0.772     |   0.9958   |  **2.46**   | **0.053** |
  | lcnet050-f256-r128-ln-cos-gap     |    0.480     |   0.9762   |    2.67     |   0.053   |
  | lcnet050-f128-r96-ln-cos          |    0.713     |   0.9944   |    2.33     |   0.029   |

  - 方法：
    - flatten: Flatten -> Linear (デフォルト)
    - gap: GlobalAveragePooling2d -> Linear
    - squeeze: Conv2d -> Flatten -> Linear
    - 解像度と特徴次元を下げる
  - squeeze 方法を使用すると、少しパフォーマンスが犠牲になりますが、モデルのサイズを半分に削減できます。
  - gap 方法を使用すると、精度が大幅に低下します。
  - 解像度と特徴次元を下げると、精度はわずかに低下します。

- **Backbone を強化**

  | 名前                             | TPR@FPR=1e-4 |    ROC     |
  | -------------------------------- | :----------: | :--------: |
  | lcnet050-f256-r128-ln-cos        |  **0.784**   | **0.9968** |
  | efficientnet_b0-f256-r128-ln-cos |    0.682     |   0.9931   |

  - パラメータ数が増えると、パフォーマンスが低下します。これは、訓練データセットの多様性が不足しているため、パラメータ数を増加させても効果が上がらないと考えられます。

- **ImageNet1K データセットの導入と CLIP モデルによる知識蒸留**

  | データセット | CLIP あり | 正規化  | クラス数  | TPR@FPR=1e-4 |    ROC     |
  | :----------: | :-------: | :-----: | :-------: | :----------: | :--------: |
  |    Indoor    |     X     |   LN    |  390,144  |    0.772     |   0.9958   |
  | ImageNet-1K  |     X     |   LN    | 1,281,833 |    0.813     |   0.9961   |
  | ImageNet-1K  |     V     |   LN    | 1,281,833 |    0.859     |   0.9982   |
  | ImageNet-1K  |     V     | LN + BN | 1,281,833 |  **0.912**   | **0.9984** |

  データセットの規模が拡大したため、従来の設定ではモデルがうまく収束しませんでした。

  そのため、いくつかの調整を行いました：

  - **設定**
    - クラス数: 1,281,833
    - エポック数: 40
    - 1 エポックあたりのデータ数: 25,600,000（モデルが収束しない場合、データ量が不足している可能性があります）
    - バッチサイズ: 1024
    - 最適化アルゴリズム: AdamW
    - 学習率: 0.001
    - 学習率スケジューラ: PolynomialDecay
    - 設定：
      - squeeze: Conv2d -> Flatten -> Linear
  - ImageNet-1K を使用してクラス数を約 130 万に拡張し、モデルに多様な画像変化を提供して、データの多様性を増加させ、パフォーマンスを 4.1% 向上させました。
  - ImageNet-1K を基に CLIP モデルを導入し、学習過程で知識蒸留を行った結果、TPR@FPR=1e-4 の比較基準でさらに 4.6% の向上を達成しました。
  - BatchNorm と LayerNorm を同時に使用すると、結果は 91.2% に向上しました。

## 評価結果

モデルの能力評価には TPR@FPR=1e-4 の基準を採用しましたが、実際にはこの基準は比較的厳格であり、デプロイ時にユーザー体験が悪化する可能性があります。したがって、デプロイ時には TPR@FPR=1e-1 または TPR@FPR=1e-2 のしきい値設定を採用することをお勧めします。

現在、デフォルトのしきい値は `TPR@FPR=1e-2` に設定されています。このしきい値は私たちのテストと評価に基づいて、適切だと考えられています。詳細なしきい値設定表は以下の通りです：

- **lcnet050_cosface_f256_r128_squeeze_imagenet_clip_20240326 結果**

  - **`model_cfg` を "20240326" に設定**
  - **TPR@FPR=1e-4: 0.912**

    |    FPR    | 1e-05 | 1e-04 | 1e-03 | 1e-02 | 1e-01 |   1   |
    | :-------: | :---: | :---: | :---: | :---: | :---: | :---: |
    |    TPR    | 0.856 | 0.912 | 0.953 | 0.980 | 0.996 |  1.0  |
    | Threshold | 0.705 | 0.682 | 0.657 | 0.626 | 0.581 | 0.359 |

  - **TSNE & PCA & ROC 曲線**

    ![result](./resources/cosface_result_squeeze_imagenet_clip_20240326.jpg)