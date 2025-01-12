---
sidebar_position: 8
---

# 関連リソース

テキスト合成ツールは、特に深層学習モデルを訓練するために大量のラベル付きデータが必要な場合に、画像データセットを自動生成するためのものです。これらのツールは、画像内に合成テキストを埋め込むことで、実世界のテキストの出現をシミュレーションし、異なる環境、フォント、色、背景へのモデルの適応性を向上させます。

以下は、現在広く知られているテキスト合成ツールの一覧です：

## SynthText

- [**SynthText**](https://github.com/ankush-me/SynthText)

SynthText はオープンソースプロジェクトで、2016 年の CVPR（コンピュータビジョンおよびパターン認識会議）で Ankush Gupta、Andrea Vedaldi、Andrew Zisserman によって発表されました。このプロジェクトは、自然画像中のテキスト位置検出研究のために、合成テキスト画像を生成することを目的としています。これらの合成画像は、特に光学文字認識やテキスト検出の分野で、機械学習モデルの訓練に使用されます。

SynthText は自然シーン画像を背景として利用し、その中に自然に見えるテキストを合成します。このプロセスには、背景画像の深度およびセグメンテーション情報が必要です。プロジェクトの主な依存関係には、pygame、opencv、PIL、numpy、matplotlib、h5py、scipy が含まれます。

## SynthText3D

- [**SynthText3D**](https://github.com/MhLiao/SynthText3D)

SynthText3D は Unreal Engine 4.16 および UnrealCV プラグインに基づいており、仮想環境内にテキストを埋め込むことで、実世界のテキスト出現をシミュレーションします。このプロジェクトは、カスタム仮想カメラとテキストレンダリング技術を使用して、テキストとその周囲環境の相互作用をキャプチャします。

プロジェクトでは、30 の異なるシーンから合成された 10K 画像のデータセットを提供しています。ICDAR 2015、ICDAR 2013、および MLT コンペティションのベンチマークテストで、SynthText3D によって合成されたデータは優れたパフォーマンスを示しました。

## UnrealText

- [**UnrealText**](https://github.com/Jyouhou/UnrealText/)

UnrealText は、3D グラフィックスエンジンを使用してシーンテキスト画像を合成する革新的なプロジェクトです。このプロジェクトは関連する学術論文と密接に関連しており、詳細は論文[**《UnrealText: Synthesizing Realistic Scene Text Images from the Unreal World》**](https://arxiv.org/abs/2003.10608)をご覧ください。

UnrealText は、強力なゲーム開発ツールである Unreal Engine を利用して、非常にリアルなテキストシーンを作成します。このプロジェクトでは、ソースコードと使用説明に加え、UE プロジェクトのデモや 30 の事前コンパイル済みシーンの実行ファイルも提供されています。また、大規模なラテン語/英語および多言語の合成シーンテキストデータセットも公開されています。

## SynthTIGER

- [**SynthTIGER**](https://github.com/clovaai/synthtiger)

SynthTIGER（Synthetic Text Image Generator）は、NAVER Corp. によって開発されたオープンソースプロジェクトで、光学文字認識（OCR）モデルの訓練を支援するために合成テキスト画像を生成します。このツールは、テキスト認識モデルのパフォーマンスを向上させるために特別に設計され、2021 年の国際文書分析および認識会議（ICDAR）で発表されました。

SynthTIGER は多言語および多様なテキストスタイル生成をサポートしており、豊富で多様な訓練データを生成できます。フォントのカスタマイズ、色のマッピング、テンプレートのカスタマイズなど、ユーザーが特定のニーズに応じて生成プロセスを調整できます。また、大規模なデータセットを複数の小さなファイルに分割して提供し、ダウンロードと使用を容易にしています。

## TextRenderer

- [**text_renderer**](https://github.com/Sanster/text_renderer)

Text Renderer は、oh-my-ocr 組織によって開発されたオープンソースプロジェクトで、深層学習 OCR モデルの訓練用のテキスト行画像を生成します。このプロジェクトはモジュール設計に重点を置いており、ユーザーは語彙、効果、レイアウトなどのさまざまな要素を簡単に追加できます。

Text Renderer は、画像拡張をサポートする imgaug を統合し、複数の語彙を画像にレンダリングして異なる効果を適用します。さらに、Text Renderer は垂直テキストの生成をサポートし、PaddleOCR と互換性のある lmdb データセットを生成できます。

## TextRecognitionDataGenerator

- [**TextRecognitionDataGenerator**](https://github.com/Belval/TextRecognitionDataGenerator)

TextRecognitionDataGenerator（TRDG）は、Belval によって開発されたオープンソースプロジェクトで、光学文字認識（OCR）ソフトウェアの訓練用にテキスト画像サンプルを生成します。このプロジェクトは非ラテン文字を含む多言語テキスト生成をサポートし、フォント、背景、テキスト効果（傾斜、歪み、ぼかしなど）を使用して画像を生成します。

TRDG は、コマンドラインインターフェイス（CLI）または Python モジュールとして使用でき、訓練パイプラインに直接統合できます。ユーザーは簡単なコマンドでランダムテキストを含む画像を生成したり、特定のテキスト内容を指定したりできます。また、垂直テキストや文字単位マスクの生成もサポートしており、テキスト画像生成の柔軟性と適用性を高めています。
