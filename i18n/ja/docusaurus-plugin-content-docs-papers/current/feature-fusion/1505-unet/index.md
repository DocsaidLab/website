---
title: "[15.05] U-Net"
authors: Zephyr
---

## 融合の序章

[**U-Net: Convolutional Networks for Biomedical Image Segmentation**](https://arxiv.org/abs/1505.04597)

---

この VGG の現世が現れてから、まだ満たされていない需要が多くあります。

研究者たちは、従来の CNN アーキテクチャが十分な細粒度を提供できず、生物医学画像のセグメンテーションの課題に対処するには不十分だと気付きました。

そこで、この作品が登場しました。この論文は画像セグメンテーションの古典的な作品です。

## 問題の定義

隣の棚の画像分類分野と比べると、みんなが ImageNet を使って楽しく過ごしている中で、生物医学画像セグメンテーションの研究者たちはそうではありません。この分野では、訓練に使えるデータが非常に少なく、深層学習の訓練に必要な量を支えるには足りません。

問題を解決する方法もはっきりしていません。以前、訓練データを複数の小さなブロックに切り分けて、より多くの訓練サンプルを作り出す方法がありました。しかし、これには別の問題があり、コンテキスト情報が失われるため、セグメンテーションの精度が低下します。

ちょうどその時、別の研究で全畳み込みネットワークアーキテクチャが提案され、著者にインスピレーションを与えました。

- [**[14.11] セマンティックセグメンテーションのための全畳み込みネットワーク**](https://arxiv.org/abs/1411.4038)

  ![fcn arch](./img/img3.jpg)

おそらく、このアーキテクチャを生物医学画像セグメンテーションの問題に適用することで、コンテキスト情報の喪失問題を解決できるかもしれません。

## 問題の解決

### モデルアーキテクチャ

![U-Net arch](./img/img1.jpg)

画像全体を使用することで、確かにコンテキスト情報の喪失問題は解決されましたが、データ不足の問題は依然として残ります。そこで著者は U-Net アーキテクチャを提案し、高解像度の特徴マップを繰り返し使用することでセグメンテーションの精度を向上させ、同時にデータ量の必要性を低減させました。

上図は U-Net のアーキテクチャで、数字の部分は一時的に無視してください。なぜなら、著者は畳み込み層でパディングを使用していないため、畳み込み層を通るたびに特徴マップのサイズが減少します。これにより、このアーキテクチャを初めて見る人は数字に惑わされて、アーキテクチャを十分に評価できなくなるかもしれません。

この図を半分に切り、左側を見てみましょう：

![U-Net arch left](./img/img4.jpg)

ここがいわゆる Backbone の部分で、この部分は自由に異なるアーキテクチャに変更できます。もし MobileNet が好きなら MobileNet を使い、ResNet が好きなら ResNet を使います。

基本的な Backbone 設計では、5 層のダウンサンプリングがあり、それぞれ上図の 5 つの出力層に対応しています。

次に右側を見てみましょう：

![U-Net arch right](./img/img5.jpg)

ここがいわゆる Neck の部分で、この部分の特徴は最下層からアップサンプリングを行うことです。方法としては単純な補間や、より複雑な逆畳み込みを使うことができます。この論文では著者は逆畳み込みを使用しています。

アップサンプリング後には、より高解像度の特徴マップが得られます。この時、最下層の特徴マップと上一層の特徴マップを融合させます。融合方法としては、単純に結合するか加算するかの方法がありますが、ここでは結合を使用しています。

上記の手順を経て、最終的には元の画像と同じサイズのセグメンテーション結果が得られます。このセグメンテーション結果は、チャンネル数によって制御できます。もし二値セグメンテーションであれば 1 チャンネルだけで済みますが、複数クラスのセグメンテーションであれば、複数のチャンネルが必要です。

:::tip
もし加算を選んだ場合、それはもう一つの古典的な作品である FPN です。

- [**[16.12] FPN: ピラミッド構造**](../1612-fpn/index.md)
  :::

## 討論

### ISBI 細胞追跡チャレンジ 2015

![isbi](./img/img2.jpg)

著者は U-Net を ISBI 2014 年および 2015 年の細胞追跡チャレンジに適用しました：

- PhC-U373 データセットでは、92％の IOU を達成し、当時の 2 位の 83％を大きく上回りました。
- DIC-HeLa データセットでは、77.5％の IOU を達成し、同じく 2 位の 46％を大きく超えました。

これらの結果は、U-Net が異なる種類の顕微鏡画像セグメンテーションタスクで卓越した性能を発揮し、既存の他の方法を大きく上回ることを示しています。

## 結論

U-Net の設計方法は、高解像度の特徴マップを保持し、コンテキスト情報を融合させることでセグメンテーションの精度を向上させ、同時にデータ量の要求を低減させました。このアーキテクチャはシンプルで拡張が容易であり、細胞分割、器官分割、病変検出など、さまざまな画像セグメンテーションタスクに適用できます。

FPN と比較すると、結合構造はより大きなパラメータ量と計算量を伴うため、パラメータ量に制限がある場合には若干の困難を招くことがあります。各アーキテクチャにはそれぞれの長所があり、いくつかの異なる設計方法を学び、タスクの要求に最も適したアーキテクチャを選択することが重要です。