---
slug: /
sidebar_position: 1
---

# 開源專案

現在你看到的這個頁面，是用來寫技術文件的。

- 如果你有興趣閱讀相關的論文分享，請前往：[**Papers**](https://docsaid.org/papers/intro)。
- 想了解更多技術心得與討論，請瀏覽：[**Blog**](https://docsaid.org/blog)。

## 📂 公開專案一覽

目前我在 Github 上公開了幾個已經完成的專案，包括：

### 工具與整合類

- [**AutoTraderX**](./autotraderx/index.md)：

  這是我練習串接臺灣證券交易券商的系統而留下的紀錄，目前只探索了「元富證券」的 API，之後預計要去探索「富邦證券」，但是還沒有安排時間。

  :::tip
  如果你要問我開發心得？那大概是心有餘悸吧。😓
  :::

  ***

- [**DocsaidKit**](./docsaidkit/index.md)：

  這是我自己寫的工具箱，裡面定義了一些在電腦視覺領域中常用的結構，例如 `Boxes`、`Polygons`。

  除此之外，還有一些影像處理（opencv）、模型架構（pytorch）和推論工具（onnxruntime）和環境配置的內容都放在這裡，這些都是我在工作中常用到的工具。

  ***

- [**GmailSummary**](./gmailsummary/index.md)：

  這是我練習串接 Gmail 和 OpenAI 而留下的紀錄，裡面的內容可能在未來 Google 和 OpenAI 的 API 更新後會失效。

  之前這個專案有運作過幾個月，但目前已經把儲值給 OpenAI 的錢花完了，所以這個專案已經停止工作。

  ***

- [**WordCanvas**](./wordcanvas/index.md)：

  之前我有陸續完成一些合成訓練資料的工具，後來覺得太散亂，所以把一些基本功能抽象出來，整合成一個新的工具，這個專案的功能主要就是把字型檔案渲染成圖像。

### 深度學習專案

- [**DocAligner**](./docaligner/index.md)：

  這是一個文件對齊的專案，功能是定位文件的四個角點。

  雖然這個功能很簡單，但是很多應用場景中都可以派上用場，目前只有定位四個角點，如果有時間我會再加上一些其他的功能。

  ***

- [**DocClassifier**](./docclassifier/index.md)：

  這是一個文件分類的專案，功能是將文件分類到不同的類別。

  這個專案有開放訓練模組，我的每個模型專案都是用相同的構建邏輯，如果你對其他的模型有興趣，可以參考這個專案，建立屬於你自己的訓練環境。

  ***

- [**MRZScanner**](./mrzscanner/index.md)：

  這個功能是辨識文件上的 MRZ 區域。

  之前想要做一個 End-to-End 的模型，雖然最後效果不如預期，但還是有一些小成果，所以我把它整理成一個開源專案，希望能夠幫助到有需要的人。

## 🚧 開發與未公開專案

除了以上公開的專案外，還有一些專案正在開發中，或是處於內部測試階段。

如果有特別感興趣的議題或想法，也歡迎與我聯繫。

## 🌍 多國語系支持

我以中文為撰寫主體，然後再進行其他語言的翻譯。

考慮到我的能力有限，沒辦法自己扛下翻譯的工作，所以我請市面上的各種 `GPTs` 來幫助我完成這件事情。

我會截取每篇文章的段落，直接交給 `GPTs` 進行翻譯。得到翻譯結果後再進行人工校對，排除一些肉眼可見的錯誤。

如果你在閱讀過程中發現了：

- **錯誤或毀損的連結**
- **錯誤的翻譯**
- **錯誤的理解**

都歡迎在文章底下留言，我會優先安排修復。

:::info
另外兩種方式，其一是到 github 上的討論區提出問題：

- [**Help Us Improve Translation Accuracy**](https://github.com/orgs/DocsaidLab/discussions/12)
- [**翻訳の正確性向上のためのご協力をお願いします**](https://github.com/orgs/DocsaidLab/discussions/13)

其二是直接發 PR 給我，我確認後可以直接合併到專案主線中，省時省力。
:::

## 🔄 調整模型

這可能是你最關心的主題了。

根據我所定義的主題，配上我提供的模型，相信能解決大部分的應用場景。

我也知道，有些場景可能需要更好的模型效果，因此必須自行搜集資料集，並且進行模型微調。

你可能在這一步就卡住了，大部分的人都是這樣，別緊張。

### 情境一

你知道我提供的專案功能符合你的需求，但你不會調整。

這種情況下，你可以直接寄封信給我，提出你的需求，然後給我「你想要解決的資料集」。我可以幫你進行模型微調，這樣你就可以得到更好的模型效果。

不用收錢，但是不能壓時間，我也不保證會執行。(這很重要！)

我雖然做的是開源專案，但也不是吃飽了撐著，當緣份到了，模型自然就會更新了，你只需要寫個郵件就「可能」得到更好的模型效果。再怎麼說，也能算是雙贏吧？

### 情境二

你想開發特定的功能，但不趕時間。

那就來信跟我討論吧，如果我覺得有趣，我會很樂意幫你開發，但我希望你可以先準備好一定規模的資料集，因為我就算我有興趣，也不一定有時間拿到足夠量的資料，或是一些特殊的資料是需要有特殊管道才能取得。

這個情境跟上述一樣，不用收錢，但是不能壓時間，我也不保證會執行。

:::tip
如果特定功能是針對那些公開的模型比賽？答案是不行。因為那些比賽多少都有版權和相關限制，如果被投訴的話，主辦單位會來找我麻煩。
:::

### 情境三

你追求快速開發的特定功能。

當時間成為你的首要考量時，我們可以轉向委任開發的合作方式，根據你的需求，我會根據我的開發時間提出一個合理的價格。

一般來說，我會保留專案的所有權，而你可以自由使用它。我不推薦買斷專案，因為這並不符合持續進步的理念。隨著技術的進步，今天的解決方案可能很快就會被更新的方法所取代。如果你買斷了一個專案，隨著時間的流逝，可能會發現這筆投資失去了其原有的價值。

:::tip
你可能會不能理解專案的所有權的歸屬問題。

仔細思考一下，說不定你只是是想「喝牛奶」而已，而不是真的想要「養一頭牛」。

- 養一頭牛多累？（要養工程師來維護專案）
- 佔空間又難照顧。（要建置訓練機器，租雲端機器很貴，買主機又容易壞）
- 怕冷又怕熱。（模型調參調到懷疑人生）
- 還一言不合就死掉。（達不到預期成果）
- 真的有夠虧。（花了錢買斷專案）
  :::

此外，大多數的專案最有價值的地方是資料集，其次才是解決方案的思考方式。在不開源私有資料集的情況下，拿到了一份程式碼的最大功能大概就是觀賞用途而已。

如果你在仔細思考過後，還是堅持要買斷專案，那我也不會攔著你，來吧。

## ✉️ 聯繫方式

如果你有任何問題，或對我的工作感興趣，歡迎隨時聯繫我！

這是我申請的工作信箱：**docsaidlab@gmail.com**，你可以寄信過來，或是直接在本網站上找一篇文章，在底下留言，我都會看到。

## 🍹 最後

除非獲得你的允許，否則在所有形式的開發專案中，我們絕對不會開源你提供的資料。資料只會用於更新模型。

感謝你的閱讀與支持，希望 **DOCSAID** 能為你帶來幫助與啟發！

＊

2024 © Zephyr