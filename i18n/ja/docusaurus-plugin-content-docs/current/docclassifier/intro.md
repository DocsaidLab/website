---
sidebar_position: 1
---

# イントロダクション

過去のプロジェクト経験から、分類モデルは最も一般的な機械学習のタスクと言えます。

分類モデルには特別難しいところはなく、まずバックボーンを構築し、最後の出力を複数の特定のカテゴリにマッピングし、最後にいくつかの評価指標を使ってモデルの良し悪しを評価します。通常、精度、再現率、F1 スコアなどが使われます。

一見、これは直線的に思えますが、実際の応用ではいくつかの問題に直面します。ここでは本プロジェクトのテーマを例に挙げてみましょう。

### カテゴリ定義

どんな分類タスクでも、カテゴリを明確かつ正確に定義することは重要です。しかし、もし定義したカテゴリ同士の類似度が非常に高ければ、モデルはこれらのカテゴリを区別するのが難しくなる可能性があります。

- 例えば：会社 A の保険書類 vs 会社 B の保険書類。

これらの二つのカテゴリはどちらも会社の書類であり、差異はほとんどないかもしれません。このため、モデルがこれらのカテゴリを区別するのが難しくなる可能性があります。

### データの不均衡

ほとんどのシーンでは、データの収集が非常に難しい問題です。特に機密情報を扱う場合、この問題は顕著です。このような場合、データの不均衡の問題に直面する可能性があります。この問題は、モデルが少数のカテゴリを予測する能力に影響を与える可能性があります。

### データ拡張

業界では、大量の書類が存在し、常に新しい書類カテゴリを追加したいと考えています。しかし、カテゴリを追加するたびに、モデル全体を再訓練または微調整する必要があり、このコストは非常に高いです。すべてのコストはさまざまな形で発生します。例えば、データ収集、ラベリング、再訓練、再評価、デプロイなど、すべてのプロセスが再度行われる必要があります。

### サブラベルのカテゴリ

顧客の要求は非常に多様です。

ある顧客が最初に一つのファイルタイプを定義したと仮定します。仮にそのファイルを A ファイルと呼ぶことにします。

その後、顧客は A ファイルについて、さらに多くのサブラベルを提供したいと言っています。例えば：

- 汚れた A ファイル
- 反射がある A ファイル
- 第一世代フォーマットの A ファイル
- 第二世代フォーマットの A ファイル
- ...

ここでは、毎回サブラベルを追加するたびにモデルを再実行しなければならないという問題は置いておいて。

モデルエンジニアリングの観点から見ると、これらのラベルを独立したカテゴリとして扱うのは「不合理」です。なぜなら、これらはすべて A ファイルに基づいています；もしこれらのラベルを多クラス問題として扱うと、それも「不合理」です。なぜなら、異なるメインフォーマットのファイルに対するサブラベルは異なるからです。

:::tip
あなたはこう考えます：問題を解決できないなら、その問題を提起した人を解決すれば良いのではないか。

- だめです！

これは機械学習の問題です。
:::

## メトリック学習

ファイル分類の問題を飛び出してみると、この問題が実際に言っていることは、**メトリック学習（Metric Learning）**のことだとわかります。

メトリック学習の主な目的は、最適な距離測定を学習することで、サンプル間の類似性を測定することです。従来の機械学習分野では、メトリック学習は通常、データを元の特徴空間から新しい特徴空間にマッピングすることを含みます。この空間では、類似したオブジェクトは近くに、異なるオブジェクトは遠くに配置されます。このプロセスは通常、サンプル間の真の類似度をよりよく反映する距離関数を学習することで実現されます。

もし前の文を読んでもまだ理解できない場合、一言でまとめると：**メトリック学習は類似性を学習する方法です**。

### 応用シーン

メトリック学習は以下の 2 つの有名な応用シーンで非常に重要です：

- **顔認識（Face Recognition）**：先程述べたような問題、顔の数は増え続け、モデルを再訓練し続けることができません。そのため、メトリック学習のアーキテクチャを使用することで、より良い距離関数を学習し、顔認識の精度を向上させることができます。

- **推薦システム（Recommendation System）**：推薦システムはユーザーの履歴に基づいて、ユーザーが興味を持ちそうな商品を推薦します。この過程で、ユーザー間の類似性を測定し、類似したユーザーの行動を見つけて、商品を推薦するために使用します。

これらの応用シーンでは、2 つのオブジェクト間の類似性を正確に測定することが、システム性能を向上させるための鍵となります。

## 問題解決

すべての分類問題がメトリック学習にまで引き上げるべきだとは言えませんが、このプロジェクトでは、メトリック学習という武器が、前述の障害を解決するのに役立ちます。

- **障害 1：カテゴリ定義**

  我々が学習する目標は、より良い距離関数であり、この距離関数が類似したカテゴリをより良く区別するのを助けます。だからこそ、もはやカテゴリを定義する必要はありません。我々が分類したい対象は、最終的にすべて一つの登録データになります。

- **障害 2：データの不均衡**

  我々はもはや大量のデータ収集を必要としません。なぜなら、我々のモデルは大量のサンプルに依存しないからです。たった 1 つのサンプル、それが我々の登録データです。他の部分は他の訓練データで訓練できます。

- **障害 3：カテゴリ拡張**

  カテゴリの拡張は、新しいデータを登録するだけで済み、モデルの再訓練は必要ありません。この設計により、訓練コストを大幅に削減できます。

- **障害 4：サブラベルのカテゴリ**

  メトリック学習のフレームワークでは、この問題も非常にうまく解決できます。サブラベルを新しい登録データとして扱うことで、元々のモデルに影響を与えることなく処理できます。サブラベルとメインラベルは特徴空間で非常に近い距離にありますが、完全に同じではないため、この 2 つのカテゴリをうまく区別できます。

---

我々はまず、メトリック学習のアーキテクチャである [**PartialFC**](https://arxiv.org/abs/2203.15565) を導入しました。このアーキテクチャは [**CosFace**](https://arxiv.org/abs/1801.09414) や [**ArcFace**](https://arxiv.org/abs/1801.07698) などの技術を組み合わせ、事前に多くの分類を設定することなく、正確な分類を行うことができます。

その後、さらに進んだ実験で [**ImageNet-1K データセット**](https://www.image-net.org/) と [**CLIP モデル**](https://arxiv.org/abs/2103.00020) を導入しました。ImageNet-1K データセットを基盤として使用し、各画像を 1 つのカテゴリとして扱うことで、分類するカテゴリ数を約 130 万に拡張し、モデルに豊富な画像の変化を与え、データの多様性を増加させました。

TPR@FPR=1e-4 の比較基準で、元々のベースラインモデルに比べて約 4.1%（77.2%->81.3%）改善しました。ImageNet-1K の基盤に CLIP モデルを追加し、訓練過程で知識蒸留を行った結果、TPR@FPR=1e-4 の比較基準でさらに 4.6%（81.3%->85.9%）改善しました。

最新の実験では、BatchNorm と LayerNorm を組み合わせることで、CLIP 蒸留モデルの基盤に対して TPR@FPR=1e-4 の効果を約 4.4%（85.9%->90.3%）向上させました。

## 最後に

テストでは、我々のモデルは、万分の一（TPR@FPR=1e-4）の誤差率で、90% を超える精度を示しました。また、カテゴリの追加時に再訓練する必要はありません。

結局のところ、人顔認識システムの仕組みをそのまま持ち込んだようなものです！

開発過程で、「本当にこれでうまくいくのか？」という声を何度も上げました。前述の通り、このプロジェクトの第一世代（第一著者）はすでに一定の効果がありましたが、まだ不安定でした。このプロジェクトが公開された時点では、第三世代モデル（第二著者）に進化しており、全体の効果も向上しており、良い結果と言えます。

以前に公開した「規則正しい」プロジェクトとは異なり、このプロジェクトは非常に面白さがあります。

したがって、このプロジェクトのアーキテクチャと実験結果を公開し、このプロジェクトがあなたに何かのインスピレーションを与え、もし本プロジェクトの設計理念から新しい応用シーンを見つけたのであれば、ぜひシェアしてください。