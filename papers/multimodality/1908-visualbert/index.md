# [19.08] VisualBERT

## 序幕上的凝視

[**VisualBERT: A Simple and Performant Baseline for Vision and Language**](https://arxiv.org/abs/1908.03557)

---

在 2015 年左右，早就有很多跨模態模型嘗試，主要都是基於 LSTM 的架構來進行相關的實驗，就如本文作者所說：「多模態的模型研究，根本就不是什麼新鮮事！」。

隨著時序來到 2017 年，Transformer 架構及其注意力機制在自然語言處理領域引起了廣泛的關注，並帶來了許多創新和突破。特別是 BERT，這一代表性的模型成功地預訓練了一個通用語言編碼器，能夠預測文本中被屏蔽的單詞。

到了 2019 年，注意力機制在多模態領域中的應用也得到了極大的發展。這促使語言與視覺的結合再次成為研究的焦點，挖掘圖像中更深層次的語義細節，涵蓋了物體、屬性、部位、空間關係、動作和意圖等語義層面。

作者受此啟發，希望能透過注意力機制捕捉影像中的隱式關係，並認為預先訓練能有效學習這些關係，根據前人的研究，作者總結出幾個現階段的問題：

## 定義問題

- **視覺與語言的複雜互動：**

  - 當前的視覺語言任務（如物體辨識、視覺字幕、視覺問答和視覺推理）都要求系統理解圖像中的詳細語義，包括對象、屬性、部分、 空間關係、
  - 動作和意圖，以及如何在語言中引用和建立這些概念。

- **統一視覺與語言的模型架構：**

  - 目前的許多模型都是針對特定的視覺語言任務而設計的，而缺乏一個可以通用於各種任務的模型。

- **預訓練的重要性：**

  - 如何有效地在視覺與語言資料上預訓練模型，以提高其在下游任務的表現。

- **理解圖像語義的挑戰：**

  - 需要捕捉和理解圖像中描述的詳細語義，並將其與文字描述相關聯。

## 解決問題

### VisualBERT 模型設計

![VisualBERT 模型架構](./img/arch_visual_bert.jpg)

1. **注意力機制：**

   - VisualBERT 的核心想法是利用 Transformer 中的注意力機制「隱式」對輸入文字的元素和輸入影像中的區域進行對齊。

2. **視覺特徵：**

   - 除了 BERT 的所有組件，VisualBERT 還引入了一組名為 F 的視覺特徵來對影像進行建模。
   - F 中的每一個特徵都對應於影像中的一個物件區域，這些物件區域是由物件偵測器導出的（可能是 Faster RCNN 或其他）。
   - F 中的每一個特徵 f 是透過以下三個特徵的總和來計算的：
     - (f_o)：代表 f 物件區域的視覺特徵表示，由卷積神經網路計算。
     - (f_s)：表示它是影像特徵的分段特徵到文字特徵。
     - (f_p)：位置特徵，當單字和物件區域之間的對齊作為輸入的一部分提供時使用，並設定為與對齊的單字相對應的位置特徵的總和。

3. **結合視覺特徵與文字特徵：**

   - 視覺特徵 F 與原始文字特徵集 E 一同傳遞至多層的 Transformer。這設計使模型能夠隱式地發現兩組輸入（文字和圖像）之間的有用對齊，進而建立新的聯合表示。

這種架構的設計允許 VisualBERT 在處理多模態任務時，能夠捕捉圖像和相應文字之間的豐富語義關係，並且可以利用 Transformer 的強大能力進行深入的表徵學習。

### 預訓練機制

VisualBERT 的預訓練過程可以細分為以下三個主要階段：

1. 與任務無關的預訓練：

   - 資料來源：

     - 假設在 COCO 資料集中，有一張照片顯示一個小男孩在公園裡和他的狗玩耍。對這張照片的五個標題可能是：
       - 小男孩在公園裡玩耍。
       - 一隻狗在草地上追球。
       - 孩子和他的寵物在戶外度過快樂時光。
       - 在陽光下，男孩和狗玩得很開心。
       - 公園裡的孩子和狗互動。

   - 掩蔽語言建模：

     - 基於前面的例子，選取第一個標題「小男孩在公園裡玩耍」作為輸入，並隨機屏蔽「玩耍」這個詞，所以輸入變成「小男孩在公園裡[MASK]」。VisualBERT 的任務是根據上下文和與文字輸入相對應的圖像（即小男孩和狗在公園裡的照片）來預測被屏蔽的詞，也就是「玩耍」。

   - 句圖預測：
     - 再以同一張照片為例，給模型兩個標題：
       - (a) 小男孩在公園裡玩耍（描述該圖像）
       - (b) 老太太在市場購物（隨機選取的不相關的標題）
     - VisualBERT 會接收這兩個標題和照片作為輸入，並需要確定哪個標題是與圖像相符的。在這個情境下，答案應該是標題 (a)。

2. 針對特定任務的預訓練：

   - 在微調 VisualBERT 至特定下游任務之前，進行這一預訓練階段是為了使模型更好地適應目標領域。這個階段主要使用帶有圖像目標的掩蔽語言建模，在特定的任務資料上進行訓練，使模型習慣於新的目標域。

3. 微調：
   - 這一步驟與 BERT 的微調策略相似。首先，會根據特定的任務引入相對應的輸入、輸出層和目標。然後，再訓練 Transformer 使其最大化在該特定任務上的表現。

綜合以上這三階段的預訓練策略，作者希望使模型更加泛化且適應於多種視覺語言任務。

## 討論

在這篇研究中，作者觀察到 VisualBERT 不僅在多種任務上都表現優異，更重要的是，其訓練策略和結構設計提供了獨特的洞察。尤其是如何融合圖像與文本信息，並在這兩者之間建立深度的語義連接。

接下來，再來探討 VisualBERT 的核心優勢、預訓練的策略選擇，以及其如何確實抓取到圖片和語言之間的細緻關聯。

### 模型表現好嗎？

- **VQA**

  - 任務描述：

    - 任務目的：對給定的圖像和問題提供正確的答案。
    - 使用的資料集：VQA 2.0，由 Goyal 等人於 2017 年提出。
    - 資料集特性：包含超過 100 萬個關於 COCO 影像的問題。

  - 模型訓練：

    - 答案的選擇：訓練模型以預測 3,129 個最常見的答案。
    - 圖像特徵來源：基於 ResNeXt 的 Faster RCNN，已在 Visual Genome 上進行預訓練。

  - 第一部分：

    - 使用與本文的方法相同的視覺特徵（這裡是指特徵維度），和物件區域建議數量（這裡是指選取影像內區域的數量）的基線模型。

  - 第二部分：

    - 展示了 VisualBERT 的模型結果。

  - 第三部分：

    - 其他不可比較方法的結果，包括使用外部問答對的方法、使用多個檢測器的方法，以及模型的集合。

  - 小結：

    - 在可以比較的基線上，效果比較好。
    - 在不能直接比較的方法中，作者認為他們提出來的的方法也不差。因為這個方法「簡單且在效能上」優於其他現有的方法。

- **VCR**

  - 任務描述：

    - VCR 包括來自 11 萬個電影場景的 29 萬個問題。
    - 這些問題主要關於視覺常識。

  - 子任務：

    - VCR 任務被劃分為兩個多重選擇子任務。
    - 分別是問題回答（Q → A）和答案論證（QA → R）。
    - 對於這兩個子任務，都有獨立的模型進行訓練。

  - 影像特徵：

    - 使用 ResNet50（由 He 等人於 2016 年提出）來提取影像特徵。
    - 利用資料集中所提供的「黃金」檢測物件框和分割。

  - 文字與影像對齊：

    - VCR 資料集提供了文本中引用的單字與物件區域之間的對齊。
    - 通過使用對應的位置特徵來匹配單字和區域，模型能夠利用此對齊。

  - 比較基準：

    - 研究者將他們的方法與基於 BERT (R2C) 構建的資料集發布的模型進行比較。
    - 同時，還與在排行榜上表現最好的單一模型 (B2T2) 進行對比。

  - 小結：

    - 精簡版的 VisualBERT w/o COCO 預訓練與 R2C 有相同的資源配比，但其效能明顯優於 R2C。
    - 使用完整版本的 VisualBERT 可進一步提高效能。

  儘管 VCR (主要涵蓋電影場景) 與 COCO 之間存在顯著的領域差異，COCO 上的預訓練對於 VCR 仍然非常有幫助。

- **NLVR2**

  - 任務描述：

    - NLVR2 專注於自然語言與圖像的聯合推理。
    - 主要挑戰包括語義多樣性、組合性以及視覺推理。
    - 資料集的任務是判定給定的自然語言描述是否正確地描述了一對影像。
    - 包含超過 10 萬個與網路影像配對的英文句子範例。

  - 分段特徵調整：

    - 在 VisualBERT 中的分段特徵機制被調整。
    - 用於指派來自不同影像的特徵，利用不同的分段特徵。

  - 影像特徵：

    - 利用 Detectron（由 Girshick 等人於 2018 年提出）的現成偵測器來獲取影像特徵。
    - 每個影像使用 144 個提案來提供特徵。

  - 小結：
    - VisualBERT 顯示出優越的表現。
    - 其中，PhBERT w/o Early Fusion 和 VisualBERT w/o COCO 預訓練在效能上明顯超越了之前的領先模型 MaxEnt。
    - 完整的 VisualBERT 更進一步擴大了其與其他模型之間的性能差距。

- **FLICKR30K**

  - 任務描述：

    - Flickr30K 資料集的主要目標是檢驗系統將字幕中的短語定位到圖像的特定物件區域的能力。
    - 給定句子的一部分或片段，系統需要選擇對應的圖像物件區域。
    - 資料集包含了 30k 個影像以及近 250k 的註釋。

  - 模型配置：

    - 基於 BAN 的設定（由 Kim et al. 在 2018 年提出）。
    - 圖像特徵使用在 Visual Genome 上預先訓練過的 Faster R-CNN 來獲得。
    - 微調時，加入了額外的注意力區塊，並使用注意力頭的平均權重來預測物件框和短語之間的對齊。
    - 系統預測時，會選擇短語中最後一個子詞中被關注最多的框作為結果。

  - 小結：
    - VisualBERT 在此任務上的表現超越了目前的領先模型 BAN。
    - 有趣的是，不使用早期融合的模型與完整的 VisualBERT 在性能上沒有顯著差異，這暗示對於此任務，較簡單或淺層的模型結構可能已足夠。

### 在這模型設計中，誰最重要？

作者探討在 VisualBERT 模型中，哪些元件或設計選擇對性能最具貢獻。

他們選擇了以下四個核心元件/策略進行消融研究：

1. 與任務無關的預訓練（C1）。
2. 早期融合，即圖像和文字特徵之間早期的互動（C2）。
3. BERT 的初始化策略（C3）。
4. 句子-影像的預測目標（C4）。

實驗結果顯示：

1. 與任務無關的預訓練（C1）是非常重要的。特別是使用配對的視覺和語言資料進行預訓練對模型的性能有顯著的提升。
2. 早期融合（C2）也證明是重要的。讓圖像和文字特徵在早期就進行互動，可以增強視覺和語言之間在多個互動層中的相互作用。
3. BERT 的初始化策略（C3）也有一定的重要性。雖然模型在沒有 BERT 預訓練權重的情況下性能下降，但這種下降不如預期的那麼明顯，認為模型在 COCO 預訓練期間也學到了很多有關紮根語言的知識。
4. 句子-影像的預測目標（C4）有一定的影響，但相對於其他元件來說，它的影響較小。

:::tip
這個結論在之後 CLIP 的實驗中驗證了第一和第二個結論，只要足夠多的資料就能幹大事。至於第三點的結論，我認為這裡可以嘗試探討 BERT 的預訓練資料和 COCO 之間是否有一定的重疊性，最後一點則是依照我自己的經驗來看，這個任務可能對模型來說太簡單，沒有對模型產生應有的監督效果。
:::

### 模型真的有看到對的地方嗎？

作者探討 VisualBERT 模型中的注意力頭是否能夠正確地將句子中的實體對應到圖像中的相應物件區域，此外，作者想了解 VisualBERT 模型的注意力頭是否能夠辨識句子中的句法關係，特別是當這些句法關係與圖像區域之間存在明確的對應關係時？

1. 實體辨識：

   - VisualBERT 的許多注意力頭具有很高的準確性，且沒有受到實體辨識的直接監督。
   - 模型的較高層在進行辨識時的精度似乎有所提高，這意味著在模型的初級層可能對於如何進行實體辨識還不那麼確定，但在後續層中模型變得越來越確定。

2. 句法基礎：

   - VisualBERT 的許多注意力頭似乎可以捕捉到句法關係，尤其是動詞與其對應的參數之間的關聯。
   - 對於各種不同的句法依賴關係，作者發現 VisualBERT 中至少有一個注意力頭的性能是明顯優於基於猜測的基線的。
   - 這意味著 VisualBERT 在無需明確句法監督的情況下，能夠隱式地辨識句法結構並對其進行對應。

### 注意力分布樣態如何？

作者探討 VisualBERT 如何在多個 Transformer 層中逐步改變其注意力分布，以更精確地對齊文字和圖像中的實體或概念。

- 注意力的細化：VisualBERT 在其連續的 Transformer 層中逐步細化文字和圖像之間的對齊。例如：參考上圖的左下角。一開始「丈夫」和「女人」兩詞可能都強烈地專注於圖像中的「女人」區域，但在模型的後續層中，這種對齊變得更加明確和正確。
- 句法對齊：VisualBERT 不僅可以根據語義對齊實體，還可以根據句法對齊它們。例如：在圖片中，「戲弄」這個詞同時專注於男人和女人，而「被」這個詞只專注於男人。
- 共指解決：VisualBERT 似乎還能夠解決語言中的共指問題，例如：「她」這個詞在圖像中被正確地對齊到「女人」。

## 結論

VisualBERT 在多種視覺語言任務上都展現了卓越的表現。這些成果不僅證明了模型的效能，更重要的是，透過其內建的注意力機制，VisualBERT 提供了一個可解釋和直觀的方式來捕獲和理解資訊。

但有一件事情，不論如何都無法迴避：

- 當人們嘗試結合物件偵測的模型時，模型的架構立刻變得非常複雜且難以使用。
- 這種過度複雜的設計可能會抑制模型在實際應用中的潛力，並增加了部署和調整的困難。

因此，將此架構進行優化和簡化絕對應該被視為後續的重要研究方向。

當然，這項工作還有許多需要進一步探索和釐清的問題。例如：對於純粹的影像任務，像是場景圖解析和情境辨識，VisualBERT 是否也能展現相同的效能？此外，是否能夠進一步擴充其能力，使之在更大的字幕資料集，例如：Visual Genome 和 Conceptual Caption 上進行預訓練？

在本研究階段，儘管有許多值得進一步探討的問題，這項研究為後續的研究者指明了後續的方向。
