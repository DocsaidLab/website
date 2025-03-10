---
title: "[22.09] FRVT-Twins"
authors: Z. Yuan
---

## 双子識別能力報告

[**FRVT: Face Recognition Verification Accuracy on Distinguishing Twins (NIST.IR.8439)**](https://nvlpubs.nist.gov/nistpubs/ir/2022/NIST.IR.8439.pdf)

---

これは論文ではなく、技術報告です。

主に NIST が FRVT（顔認識ベンダーテスト）において、双子を識別する問題に関する研究を行った結果を示しています。

これは非常に興味深い問題です。なぜなら、双子の間の類似性が非常に高く、顔認識システムにとっては困難を引き起こす可能性があるからです…

あ、違いました。

正確には、顔認識システムにとって「双子を識別すること」は非常に息苦しいほど難しい課題です。

:::tip
NIST に関する情報は、[**NIST & FRVT**](https://docsaid.org/ja/blog/nist-and-frvt/)をご参照ください。
:::

## はじめに

顔認識技術は、公共および民間部門でますます広く使用されており、主に本人確認、取引認証、およびアクセス管理に利用されています。

過去 10 年間で、FRVT 評価の精度は大幅に向上し、これらの用途を支えています。

COVID-19 のパンデミック以降、一部のアルゴリズムはマスクを着用した個人を識別できるようになりました。

しかし、これらの進歩にもかかわらず、双子の識別には依然として問題があります。

本報告書は、この分野での最新のアルゴリズムの結果を記録しています。

## データセット

### Twins Day Dataset 2010-2018

- [**ダウンロード**](https://biic.wvu.edu/data-sets/twins-day-dataset-2010-1015)

このデータセットは、ウェストバージニア大学の生体認証および認識革新センターから提供され、2010 年から 2018 年までの Twins Days の画像が含まれています。

これらの収集は IRB（機関審査委員会）の協定に基づいて行われ、すべての参加者からインフォームド・コンセントを得ています。

毎年収集される画像のサイズは異なります：

- 2010 年：2848x4288
- 2011 年：3744x5616
- 2012 年から 2018 年：3300x4400 および 2400x3200。

画像はすべて高品質の正面写真で、NIST SAP50 規格（頭部と肩部の構図要件）および SAP 51 規格（頭部構図要件）を満たしています。

- **サンプル画像：異なる人物**

  ![different](./img/img1.jpg)

- **サンプル画像：同一人物**

  ![same](./img/img2.jpg)

### 移民関連画像

このデータセットには、主にウェブカメラ画像と一部のビザ申請画像からなる、リアルタイムで撮影された画像が含まれています。

ウェブカメラ画像は、サービススタッフが時間制限内で撮影したため、ピッチ角、ヨー角、およびロール角に変化があり、背景照明が明るすぎて顔の露出が不十分な場合もあります。また、近距離で撮影されたため、透視歪みが発生し、一部の顔が切り取られています。これらの画像の平均瞳距離は 38 ピクセルで、ISO/IEC 19794-5 全正面画像タイプには適合していません。

その中には、面接環境で専用機器と照明を使用して収集されたビザ申請画像も含まれており、これらの画像は ISO/IEC 19794-5 標準に準拠し、撮影サイズは 300x300、姿勢はほぼ正面です。

- **イメージサンプル**

  ![img3](./img/img3.jpg)

## データセットの欠陥

### 小規模な集団

- Twins Day データセットは、他の NIST FRVT 1:1 データセットと比較して小規模で、画像は 5900 枚以上です。
- 一部のデータのメタデータが一致しないか欠落しているため、多くの画像や双子が除外されており、誤認識の事例もあります（例えば、A と B は双子だが、B と C も双子として記録されている場合）。
- 移民関連データセットには、152 組の双子と 2478 枚の画像が含まれており、大部分の画像は保持されています。

### 識別コードの変更

- Twins Day では通常、同じ参加者に毎年同じ識別コードが割り当てられますが、時々新しい識別コードに変更されることがあります。
- これにより、双子でない比較の中で、少数の比較スコアが閾値を超え、実際には正しいマッチングであるべきものが誤りとして表示されることがあります。

### 不正確/欠落したメタデータ

- 2011 年から 2016 年にかけて、収集された双子の日データの一部またはすべてのメタデータが失われています。
- 不正確な双子の種類、誕生日などのメタデータの不一致が、多くの画像を除外する原因となっています。
- 移民関連の画像データでは、異卵双生児と一卵双生児の区別に関する情報が欠如しています。

### データの不均衡

- Twins Day の画像は、双子の種類の分布が不均衡であり、2.8%が異性愛者の異卵双生児、6.7%が同性異卵双生児、90.5%が一卵双生児です。
- データセットの年齢層の分布も不均衡であり、大部分の画像は 20〜39 歳の範囲に属し、40〜59 歳および 60 歳以上の高齢層の画像は非常に少ないです。

### 人種の不均衡

- Twins Days データセットには参加者の人種が含まれており、85%が白人、10%がアフリカ系アメリカ人、5%がその他です。
- 人種の数が多すぎるため、人種に基づいた分析にはあまり意味がありません。

## アルゴリズム分析レポート

FRVT の活動は世界中に開放されているため、多くのアルゴリズム参加者があります。

この報告書では、2019 年から 2022 年 2 月中旬にかけて FRVT 1:1 テストに提出された 478 個のアルゴリズム結果を記録しています。

### 分析指標

- **誤マッチ率 (False Match Rate, FMR)**

  誤マッチ率（FMR）は、すべてのマッチング試行の中で、異なる人物の顔が誤って同じ人物として認識される割合です。

  顔認識システムにおいては、特に高いセキュリティが求められる用途（銀行や空港のセキュリティチェックなど）では重要な性能指標です。

  FMR = 0.0001 に設定した場合、10,000 回のマッチング試行で 1 回だけ誤マッチングが発生することを意味します。

- **誤非一致率 (False Non-Match Rate, FNMR)**

  誤非一致率（FNMR）は、すべてのマッチング試行の中で、同一人物の顔が異なる人物として認識される割合です。

  これも顔認識システムの性能を測る重要な指標であり、特にユーザーの利便性やユーザー体験を重視する場合に重要です。

- **FNMR @ FMR = 0.0001**

  「FNMR @ FMR = 0.0001」のような表現は、誤マッチ率を 0.0001 に設定した場合に観察される非一致率（FNMR）の値を意味します。

  これは、顔認識システムが非常に低い誤マッチ率でどの程度パフォーマンスを発揮するかを測定するもので、非常に厳しい誤マッチ条件下でも同一人物を効果的に認識できることを保証します。

### 一卵双生児

- Twins Day データでは、一卵双生児の FMR 値は通常 0.99 以上で、これらの双子はほとんど常にアルゴリズムによって誤って互いに認識されます。
- FMR が 0.99 未満の場合は主に 2 つのケースから発生します：

  - **システムエラー**：高い登録失敗率（FTE）により、多くの画像またはすべての画像から特徴を抽出できないアルゴリズム。
  - **アルゴリズムの問題**：システムエラーによる不一致率（FNMR）が高く、その結果、低いスコアが出るアルゴリズム、つまりアルゴリズムがどれも異なる人として認識するもの。

### 一卵双生児を識別できる少数のアルゴリズム

一部のアルゴリズムは、一卵双生児の識別において高い精度を示し、これらのアルゴリズムは、すべての一卵双生児を誤って同じ人物として識別しませんでした。

これらのアルゴリズムは以下の通りです：

- **アルゴリズム**: aigen-001, aigen-002, beyneai-000, glory-004, mobai-000, iqface-001。
- **性能指標**:
  - **FNMR（誤非一致率）**: ≤ 0.02
  - **FTE（登録失敗率）**: ≤ 0.02
  - **FMR（誤マッチ率）**: ≤ 0.7

その中でも、aigen-002 の性能は特に注目に値します。このアルゴリズムの一卵双生児の誤マッチ率は 0.475 で、約半数のケースで一卵双生児を誤って同じ人物として認識しませんでした。この FMR 値は理想的な基準値（FMR = 0.0001）よりもはるかに高いですが、他のほとんどのアルゴリズムの 0.98 や 0.99 の FMR 値と比較するとかなり低く、一卵双生児の識別において優れた性能を示しています。

## 総合分析

![result](./img/img4.jpg)

### スコア分布

上図は、2 つの正確で代表的なアルゴリズムによる、異なるタイプの写真に対するスコアの分布を示しています。

その中で最も高いスコアは、「mated twins」ラベルのグループから来ており、これは同一人物が同じ日に撮影された写真間での比較を示します。

次に高いスコアは、「mated mugshot」の一般的な肖像写真から来ており、対照群として使用されています。

一卵双生児間の比較（「non-mated identical twins」とラベル付けされたもの）も、同様に高いスコアを獲得しています。

異性愛者の異卵双生児（「non-mated fraternal samesex twins」とラベル付けされたもの）のスコアも、上記の高いスコアに近いです。

これらのスコアはすべて、報告書で設定された閾値を上回っており、これは関連性のない人物間の画像ペアに対して誤認識率 FMR=0.0001 を示すように設計されています。

:::tip
双子のスコアは、同一人物のスコアに若干劣りますが、これらの結果は依然としてモデルの閾値を大きく上回っており、アルゴリズムが双子と非双子の識別において直面する挑戦を示しています。

また、報告書は、単に閾値を上げて双子間の誤認識を減らすことは効果的でないと指摘しています。問題はアルゴリズムの基本設計にあり、閾値の設定に依存していないことがわかります。
:::

### 年齢の影響

異なる年齢層の双子を分析した結果、2 つのデータセットともに、年齢層が高くなるにつれて、関連性のない双子間の類似度スコアが低下することが示されました。

**Twins Days データセット**では、双子を 4 つの年齢層に分けています：0〜19 歳、20〜39 歳、40〜59 歳、60 歳以上。分析結果は、最年長の年齢層において、異卵および同卵の同性双生児が最若年層の双子よりも低い類似度スコアを示していることを示しています。

**移民関連データセット**では、すべての同性双子が 0〜19 歳、20〜39 歳、40〜59 歳の 3 つの年齢層に分けられています。このデータセットでは、0〜19 歳の類似度スコアが関連付けられた双子のスコアに近く、40〜59 歳の類似度スコアは若年層に比べて著しく低く、関連付けられた双子のスコアと大きな差があります。

年齢層が高いほど類似度スコアが低下しましたが、これらのスコアは依然としてアルゴリズムで設定された閾値を上回っており、アルゴリズムは依然として非関連の双子をマッチングとして認識しています。したがって、アルゴリズムは異卵双生児と同卵双生児を効果的に区別できていないことが示されています。

:::tip
異卵双生児を区別できていないことは特に驚くべきことです。異卵双生児は異なる遺伝子を持っており、顔の特徴に違いがあるはずです。これは、特に同じ年に生まれた兄弟姉妹を識別する際に、アルゴリズムが正確に区別できない可能性があることを意味します。
:::

### 長期的影響

顔認識アルゴリズムの性能が向上しているにもかかわらず、同性双生児の識別能力には改善が見られませんでした。つまり、顔画像データにおいて良好なパフォーマンスを示すアルゴリズムでも、Twins Days データにおける同性異卵双生児と同卵双生児の比較においては、性能が改善されないか、逆に低下しています。

## 結論

NIST の顔認識検証テスト（FRVT）における性能と、同卵および異卵双生児の識別能力を考慮した場合、現在の顔認識技術は、特に遺伝的に非常に似ている同卵双生児を区別する際に、重大な課題に直面しています。

異卵双生児間の遺伝的類似性が低く、外見の違いが大きい可能性があるにもかかわらず、アルゴリズムはその識別においても依然として困難を抱えており、特に本人確認の精度が高く求められるシーンでは、これが重大な問題を引き起こす可能性があります。

今後の改良に向けて、報告書は以下の要点を提案しています：顔認識システムのパフォーマンスを向上させるために、より高解像度の画像を使用すること。

### 高解像度画像の活用

高解像度画像は、皮膚のテクスチャや孔のパターンなど、顔の細部を提供でき、これらは個人固有の特徴であり、遺伝的に同一の双子でも区別する手助けになる可能性があります。

:::tip
このアプローチの有効性は、2004 年の特許アルゴリズム（米国特許：US7369685B2）によって証明されています。

このアルゴリズムは、高解像度画像で見える皮膚のテクスチャを分析することで、双子を正確に区別できます。
:::

この目標を達成するために、将来の研究および開発は、少なくとも 120 ピクセルの眼間距離を持ち、ISO 基準の正面肖像に準拠した画像取得の質を向上させることに集中するべきです。さらに、現在主流の神経ネットワークモデルは、計算速度を向上させるために入力画像を直接ダウンサンプリングすることが多いですが、これにより重要な詳細情報が失われ、識別精度に影響を与えます（特に双子のような非常に似ている場合に）。

これらは今後の改善方向であり、双子識別の問題に関して、将来の FRVT でより良いパフォーマンスが期待されます。
