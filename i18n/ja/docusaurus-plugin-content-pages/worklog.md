# 作業日誌

---

:::info
このページは私たちの作業内容を記録するためのものです。
:::

---

## 2024

### 12 月

- 4 人の作成者を招待して、開発ルーチンに参加して共有します。
- **StockAnalysis Demo**の機能を完成させ、ウェブサイトにデプロイしました：[**遊楽場**](https://docsaid.org/playground/stock-demo)
- 論文ノートに「著者欄」を追加し、ホームページ、論文ノート、ブログのスタイルを美化。
- ウェブサイトのバックエンド管理システム、データベースシステム、会員登録システムの構築を開始。
- 論文ノートを執筆中、現在累計 150 本。

### 11 月

- 新たに多言語対応：[**日本語**](https://docsaid.org/ja/)
- **@docusaurus/core@3.6.1** を更新したら、また壊れた！
  - エラー部分を修正するために少し時間を割きました。
- 論文ノートを執筆中、現在累計 135 本。
- **DocumentAI**：開発を継続中。
- **TextRecognizer**：10 月の進捗を引き継ぎ、開発を続行中。

### 10 月

- **TextRecognizer**：5 月の進捗を引き継ぎ、開発を続行中。
- モデル Demo の機能を完成させ、ウェブサイトにデプロイ：[**遊び場**](https://docsaid.org/playground/docaligner-demo)
- NextCloud を自分のサーバーから GCP に移行し、すべてのダウンロードリンクを更新。

### 9 月

- **MRZScanner**：開発完了、オープンソースプロジェクトとして公開。🎉 🎉 🎉
- **TextDetector**：3 月の進捗を引き継ぎ、開発を続行中。
- 素晴らしいウェブサイトを発見し、忘れないようにメモ：
  - [**Hello アルゴリズム**](https://www.hello-algo.com/)
- 論文ノートを執筆中、現在累計 100 本。

### 8 月

- **MRZScanner**：デプロイテスト後、再調整。
- **@docusaurus/core@3.5.2** を更新したら、後方互換性がなくて驚き……
  - エラー部分を修正するために少し時間を割きました。
- OpenCV の依存関係問題を排除する決心をし、同じ問題に悩む人がいることを発見：
  - [**OpenCV 依存関係エラー修正ツール：OpenCV Fixer**](https://soulteary.com/2024/01/07/fix-opencv-dependency-errors-opencv-fixer.html)
  - オープンソースプロジェクト：[**soulteary/opencv-fixer**](https://github.com/soulteary/opencv-fixer/tree/main)
  - 感謝：[**蘇洋ブログ**](https://soulteary.com/) のシェア、時間を大いに節約できました。
- 論文ノートを執筆中、現在累計 90 本。

### 7 月

- 論文ノートを執筆、累計 80 本。
- **MRZScanner**：開発を開始。

### 6 月

- **AutoTraderX**：元富証券 API の接続を完了し、オープンソースプロジェクトとして公開。🎉 🎉 🎉
- OpenAI のクレジットが尽きたため、GmailSummary のデイリーニュース配信機能を停止。
- 論文ノートを執筆、累計 50 本。

### 5 月

- **Text Recognizer** モデルを完成。
  - 最終評価結果は良好だが、「過学習ではないふりをした過学習モデル」だと認識。（？？？）
  - 理想のアーキテクチャにはまだ距離があるため、公開は一時見送り。
- Docusaurus の Search 機能を探求し、Algolia 検索サービスをテストして導入。
  - 感謝：[**WeiWei**](https://github.com/WeiYun0912) さんの執筆記事：
    - [**[docusaurus] Docusaurus における Algolia 検索機能の実装**](https://wei-docusaurus-vercel.vercel.app/docs/Docusaurus/Algolia)
- **Text Recognizer** モデルの開発、パラメータ調整とモデル訓練を実施。
- **AutoTraderX**：開発を開始。

### 4 月

- CSS スタイルの設定を学び、ブログの外観を調整。
  - 感謝：[**朝八晚八**](https://from8to8.com/) さんの記事：[**ブログホームページ**](https://from8to8.com/docs/Website/blog/blog_homepage/)
- **TextRecognizer**：WordCanvas の進捗を引き継ぎ、文字認識プロジェクトを推進。
- **GmailSummary**：機能を変更。デイリーニュースを技術文書ページにプッシュ。
- 現在のすべてのプロジェクトの技術文書を完成。
- Docusaurus の i18n 機能を探求し、英語の文書を作成。
- Docusaurus の技術文書機能を探求し、技術文書の執筆を開始。GitHub から内容を移行。
- **WordCanvas**：開発完了、オープンソースプロジェクトとして公開。🎉 🎉 🎉

### 3 月

ある日、Google Drive のファイルダウンロード機能が壊れ、`gen_download_cmd` で取得していたデータが「エラー HTML の塊」になってしまった。👻 👻 👻

熟考の末…

最終的に [**NextCloud**](https://github.com/nextcloud) のオープンソースフレームワークを利用してプライベートクラウドを構築し、データ保存専用とすることを決定。過去のダウンロードリンクをすべて更新。

- **GmailSummary**：開発完了、オープンソースプロジェクトとして公開。🎉 🎉 🎉
- **DocClassifier**：複数の正規化層を重ねることでモデル性能が大幅に向上することを発見。（偶然の発見）
- **TextRecognizer**：初期プロジェクトの計画。
- **WordCanvas**：開発を開始。
- **TextDetector**：多くの困難に直面し、一時中断。

### 2 月

- **TextDetector**：公開データの収集。
- **DocClassifier**：CLIP を導入し、モデルの知識蒸留を実施。結果は良好！
- Docusaurus のコメント機能を探求し、giscus コメントサービスを追加。
  - 感謝：[**不務正業のアーキテクト**](https://ouch1978.github.io/) さんの記事：
    - [**Docusaurus の記事下部に giscus コメントエリアを追加する**](https://ouch1978.github.io/docs/docusaurus/customization/add-giscus-to-docusaurus)

### 1 月

- **TextDetector**：初期プロジェクトの計画。
- **DocClassifier**：開発完了、オープンソースプロジェクトとして公開。🎉 🎉 🎉

## 2023

### 12 月

- **DocClassifier**：開発を開始。
- **DocAligner**：開発完了、オープンソースプロジェクトとして公開。🎉 🎉 🎉
- **Website**：面白い Meta のオープンソースプロジェクト [**docusaurus**](https://github.com/facebook/docusaurus) を発見。これは静的ウェブサイトを簡単に作成できるツールで、Markdown を使用してコンテンツを記述できます。そのため、これを使ってブログを作ることを決定。
- WordPress で構築していたウェブサイトを削除し、その内容を GitHub 上の `website` プロジェクトに移行。

### 11 月

- **DocClassifier**：初期プロジェクトの計画。
- **DocsaidKit**：開発完了、オープンソースプロジェクトとして公開。🎉 🎉 🎉
- 論文ノートを執筆、累計 20 本。

### 10 月

- **WordCanvas**：初期プロジェクトの計画。
- **DocGenerator**：第二段階の開発が完了。テキスト合成モジュールを WordCanvas プロジェクトに分割。

### 9 月

- **DocAligner**：開発を開始。
- **DocGenerator**：第一段階の開発が完了。
- 論文ノートを執筆、累計 5 本。

### 8 月

- **DocAligner**：初期計画。
- **DocsaidKit**：よく使うツールを整理し、開発を開始。
- [**WordPress**](https://wordpress.org/) の機能を探求。自分でサイトを構築し、ブログを執筆する試み。
  - 感謝：[**諾特斯サイト**](https://notesstartup.com/) の惜しみないシェア、多くのことを学びました。
- DOCSAID の GitHub アカウントを作成し、いくつかのプロジェクトの計画を開始。

### それ以前

私たちはさまざまな仕事の間を転々とし、日々を繰り返していました。同じ夢を語る異なるボスの言葉を聞き、味気ない希望を嚙みしめていました。

幾多の夜を徹して仕上げたプロジェクトは、熱意に満ちた理想を織り交ぜていましたが、資本市場の「愛されるか、愛されないか」の狭間に揺れていました。

「愛されない」となれば、それで終わり。

青春が終わる前に、私たちは何かを残したいと考えました。

何でもいい。ここに、私たちがかつてここにいたことを記録しておきたい。
