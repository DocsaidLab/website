# [21.04] Soft Prompts

## 小弦切切如私語

[**The Power of Scale for Parameter-Efficient Prompt Tuning**](https://arxiv.org/abs/2104.08691)

---

我們剛看完 Prefix-Tuning 不久，現在來看另外一個新的方法：**Prompt Tuning**。

:::tip
如果你還沒看過 Prefix-Tuning，不妨先去看一下我們之前讀的論文：

- [**[21.01] Prefix-Tuning: 是他？不是他？**](../2101-prefix-tuning/index.md)
  :::

## 定義問題

作者在論文中從 T5 的論文架構開始，來幫助讀者了解目前在調整模型上所面臨的困難。

:::tip
如果你沒看過 T5，可以參考以下論文：

- [**[19.10] Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**](https://arxiv.org/abs/1910.10683)
  :::

T5 將所有任務視為文本生成問題，無論是翻譯、摘要還是分類，都可以表示為從輸入文本生成輸出文本。

傳統的分類模型使用概率 $\text{Pr}(y|X)$，將輸入 $X$ 映射到輸出類別 $y$，然而，在 T5 的架構中，模型所關注的是**條件生成**問題，目標是計算：

$$
\text{Pr}_\theta(Y | X)
$$

其中：

- $Y$ 是代表類別的**文本**，例如：「positive」或「negative」，而不是 0, 1, 2 這種類別。
- $\theta$ 是 Transformer 模型的參數。

這種方法的優點是模型可以直接生成更豐富的文本輸出，而不僅僅是單一的類別標籤。

提示（Prompting）是添加在輸入 $X$ 前的一段文本，用於引導模型生成正確的輸出 $Y$。提示為模型提供了任務的上下文或指令，幫助模型理解應該如何處理輸入。

例如在情感分析任務中，我們可以使用以下提示：

```
"請判斷以下句子的情感傾向："
```

然後將用戶的輸入句子接在提示後面，形成模型的最終輸入：

```
"請判斷以下句子的情感傾向：I love this movie!"
```

但是這種「傳統上」的提示，不僅人工成本高，而且效果不穩定又不可微分。不可微分的問題意味著我們無法通過反向傳播算法來更新提示的參數，這將導致模型無法自動學習到最佳的提示方式。

---

為了解決這個問題，之前的研究：Prefix-Tuning 提出在模型的輸入最前面加上一個叫做 Prefix 的 Token，來引導模型產生結果。這個 Prefix 必須深入到模型的每一層，這樣才能讓模型在每一層都能受到引導。

那我們能不能把事情簡化一下？只要在輸入層對模型進行引導就好了？

所以就有了這篇論文：**Prompt Tuning**。

:::tip
這裡說的的「引導」和另外我們常聽到的「Prompt Engineering」不是同一件事情，Prompt Engineering 還是使用自然語言的方式來引導模型，從頭到尾都沒有改變模型的輸入特徵，沒有改參數，也沒有改架構。

而「Prompt Tuning」指的是在模型的輸入層加上一個特殊的 Token，這個 Token 是可以訓練的，這樣就可以讓模型在訓練的時候自己學習到最適合的引導方式。
:::

## 解決問題

### Prompt Tuning

<div align="center">
<figure style={{"width": "80%"}}>
![model arch](./img/img1.jpg)
</figure>
</div>

所以作者提出了 Prompt Tuning 的方法，引入了可更新的提示詞嵌入參數 $\theta_P$，這些參數不再受限於模型的詞嵌入表，可以通過訓練數據自動學習。

新的條件生成公式為：

$$
\text{Pr}_{\theta; \theta_P}(Y | [P; X])
$$

其中：

- $[P; X]$ 表示將提示 $P$ 和輸入 $X$ 進行拼接。
- $\theta$ 是凍結的模型參數。
- $\theta_P$ 是可訓練的提示參數。

在訓練過程中，使用反向傳播算法，只更新提示參數 $\theta_P$，而保持模型主體的參數 $\theta$ 不變。

具體的實現細節大致上可以分為幾個步驟：

1. **輸入嵌入**：將輸入的 $n$ 個 token 嵌入為矩陣 $X_e \in \mathbb{R}^{n \times e}$，其中 $e$ 是嵌入維度。
2. **提示嵌入**：提示詞嵌入為矩陣 $P_e \in \mathbb{R}^{p \times e}$，其中 $p$ 是提示的長度。
3. **拼接操作**：將提示嵌入和輸入嵌入拼接為：
   $$
   [P_e; X_e] \in \mathbb{R}^{(p + n) \times e}
   $$
4. **模型處理**：將拼接後的嵌入輸入到編碼器-解碼器架構中進行計算。

假設我們希望模型判斷句子 "I love this movie!" 的情感，基於傳統方法的模型輸入是：

```
"請判斷以下句子的情感：I love this movie!"
```

而使用 Prompt Tuning 方法的模型輸入是：

```
[Token1] [Token2] [Token3] [Token4] [Token5] I love this movie!
```

其中，上面的每一個 Token 都是可以訓練的，模型必須自己學習如何最好地使用這些提示。

:::tip
**這不就是 AutoPrompt？**

如果你之前有讀過 AutoPrompt 的話，你可能就會問出這個問題，如果沒看過，可以參考我們之前的文章：

- [**[20.10] AutoPrompt: 模型語**](../2010-autoprompt/index.md)

---

這兩者的差別在於：在 AutoPrompt 中，模型要從現有的詞彙表中找到最適合的提示，最後的結果都存在於詞彙表中。而在 Prompt Tuning 中，提示的輸入直接作用於特徵空間，不需要受限於詞彙表。

所以基於 Prompt Tuning 的結果，你只能透過計算餘弦相似度來看這和哪些詞彙比較接近，而不能直接看到提示的內容。
:::

## 討論

![ablation](./img/img2.jpg)

作者做了一系列的消融研究，來探討 Prompt Tuning 的一些關鍵問題：

### 提示長度要多長？

如上圖 (a)，作者對不同模型規模（Small、Base、Large、XL、XXL）的提示長度進行了實驗，提示長度分別為 \{1, 5, 20, 100, 150\}。

結果顯示，對大多數模型而言，提示長度增加到多於 1 個 token 對性能表現有明顯提升。但對於 T5-XXL 模型，即使只使用單一 token 作為提示，也能達到不錯的表現，表明大模型對提示訊號的需求較低。

超過 20 個 token 後，性能提升趨於平緩，只帶來微小的增益。

### 提示初始化策略的影響？

如上圖 (b)，作者比較了三種不同的初始化策略：

1. **隨機初始化**：從範圍 $[-0.5, 0.5]$ 中均勻隨機取樣。
2. **詞彙嵌入初始化**：從 T5 的 **5,000 個最常見詞彙**中選取詞嵌入進行初始化。
3. **類別標籤初始化**：將下游任務中的類別標籤轉換為詞嵌入，若標籤為多個 token，則對嵌入取平均值。當提示長度超過類別數時，剩餘的 token 使用詞彙嵌入填充。

結果顯示，類別標籤初始化在所有模型規模下表現最佳，特別是對於小模型來說，初始化策略的差異非常明顯。T5-XXL 模型對初始化策略較不敏感，無論使用何種初始化，其性能都相對穩定。

### 預訓練目標的影響？

如上圖 (c)，作者探討了不同的預訓練目標對 Prompt Tuning 的影響：

1. **Span Corruption**：使用預設的 T5 span corruption 預訓練目標。
2. **Span Corruption + Sentinel**：在下游任務的目標輸出中添加哨兵符號，以模擬預訓練時的輸出格式。
3. **LM 調適**：延續 T5 的預訓練，但改為**語言模型（LM）目標**，進行額外的 100,000 步調適。

結果顯示，Span Corruption 預訓練的模型不適合用於凍結模型的 Prompt Tuning，因為模型習慣了讀取和輸出帶有哨兵符號的文本。即使透過 **「Span Corruption + Sentinel」** 模擬預訓練格式，效果仍然有限。

**LM Adaptation** 在所有模型規模上都顯著提升性能。

### LM 調適時長的影響？

如上圖 (d)，作者探討了 LM 調適時長對 Prompt Tuning 的影響。

結果顯示，延長 LM 調適步數會帶來額外的增益，並在 100,000 步左右達到最佳效果。Span Corruption 預訓練轉換為 LM 目標不是一個簡單的過程，需要投入相當的訓練資源（相當於原始 T5 預訓練步數的 10%）。

T5-XXL 在各種非理想配置下仍表現良好，顯示其對模型設定具有高韌性。在 Span Corruption 配置下，模型表現不穩定，小模型甚至超越了 Base、Large 和 XL 模型。這些問題並非隨機波動造成，因為在 3 次重複實驗中觀察到一致的低變異。

與 Span Corruption 預訓練的模型相比，LM 調適後的模型在所有規模下都表現穩定，大幅降低了性能不穩定的風險。

### 和其他方法比較

<div align="center">
<figure style={{"width": "80%"}}>
![comparison](./img/img3.jpg)
</figure>
</div>

上圖中，作者將 Prompt Tuning 與其他相關方法進行了比較，由於方法眾多，下面會列出每個方法的簡要介紹和參考文獻。

---

- **Prefix Tuning**：

  - [**[21.01] Prefix-Tuning: Optimizing Continuous Prompts for Generation**](https://arxiv.org/abs/2101.00190)

  在 Transformer 的每一層前置可學習的前綴 (prefix)，相當於為每層網路固定激活值。這方法適用於 GPT-2 和 BART，而本研究的 Prompt Tuning 專注於 T5。在 BART 上，Prefix Tuning 需要同時在「編碼器和解碼器」加入前綴，而 Prompt Tuning 只需在編碼器加入提示。

  Prompt Tuning 只需在輸入層加上一個單一提示詞，而非在每層加入前綴，因此參數需求更少。而且 Prompt Tuning 允許 Transformer 根據輸入例子更新其中間層的任務表徵，而 Prefix Tuning 需要重參數化來穩定訓練。

---

- **WARP**

  - [**[21.01] WARP: Word-level Adversarial ReProgramming**](https://arxiv.org/abs/2101.00121)

  WARP 將提示參數添加至輸入層，並使用 [MASK] token 和可學習的輸出層，將遮蔽部分映射至類別預測。這種方法只能產生單一輸出，因此受限於分類任務。

  Prompt Tuning 不需要對輸入進行特殊設計或使用任務專屬的輸出層，適用於更廣泛的任務。性能也更接近完整的模型微調。

---

- **P-tuning**

  - [**[21.03] GPT Understands, Too**](https://arxiv.org/abs/2103.10385)

  P-tuning 將可學習的連續提示嵌入於輸入之間，並基於人類設計的模式進行排列。為了達到良好的 SuperGLUE 表現，P-tuning 必須與模型微調結合，即同時調整提示和主模型的參數。

  Pompt Tuning 只需更新提示參數，而主語言模型保持凍結，避免模型微調的成本。

---

- **Soft Words**

  - [**[21.04] Learning How to Ask: Querying LMs with Mixtures of Soft Prompts**](https://arxiv.org/abs/2104.06599)

    Soft Words 學習的提示基於手工設計的提示範本，並為每層添加可學習的 $\Delta_i$ 參數，使得參數需求隨模型深度增加。

    Pompt Tuning 不需要隨層數增加而添加額外參數，因此在參數規模上更具效率。

---

- **Adapters**

  - [**[19.02] Parameter-Efficient Transfer Learning for NLP**](https://arxiv.org/abs/1902.00751)

    Adapters 是插入於凍結模型層之間的小型瓶頸層，用以減少任務專屬參數。在 BERT-Large 上微調 Adapter 層，僅增加 2–4% 的參數，且性能接近完整模型微調。

    Adapters 透過重寫中間層的激活值來修改模型行為，而 Pompt Tuning 則是透過調整輸入表示，保留了模型內部的運算不變。

### 到底提示了什麼？

如同剛才講到的，由於 Prompt Tuning 是在連續空間中操作，而非明確的詞彙空間，我們難以直接理解這些提示是如何影響模型的行為。

作者透過計算每個提示 token 與模型詞彙表中各 token 的「餘弦相似度」，找出最相近的詞彙。這讓我們可以看到每個提示 token 對應的「最近鄰詞彙」，藉此找到提示 token 在語義上的含義。

實驗結果顯示提示 token 的前五個最近鄰詞彙往往形成「語義密切相關的群組」。當使用隨機生成的向量代替經過訓練的提示 token，則無法形成相似的語義群聚。

這意味著 Prompt Tuning 不是隨機的，而是確實能夠捕捉到語言模型中的語義結構，此外在長提示序列的情況下，作者發現多個提示 token 可能共享相同的最近鄰詞彙。

但這又衍生出兩個潛在問題：

1. **冗餘容量**：提示中可能存在重複或多餘的資訊，無法進一步提升模型效能。
2. **缺乏序列結構**：提示 token 的表徵未能精確反映序列中的位置資訊，導致模型難以準確地定位和解析關鍵訊息。

另一個重要的觀察是，提示 token 的最近鄰詞彙中，經常包含「下游任務的類別標籤」，意思是 Prompt Tuning 能在模型內部儲存預期的輸出類別，作為生成輸出的參考依據。

## 結論

Prompt Tuning 在各類實驗中的表現與傳統模型微調相當，且隨著模型規模的擴大，這種性能差距逐漸縮小。在零樣本領域遷移任務中，Prompt Tuning 展現出更好的泛化能力，表示凍結語言模型中的通用理解參數並將學習範圍限制於輕量化的提示向量，可以有效避免過度擬合於特定領域。

作者認為未來的研究方向，可能在於將「任務定義的參數」與「語言建模的通用參數分離」，這樣可以更好地控制模型的行為，並提高模型的可解釋性。

:::tip
後續的有不少的研究也都在 Prompt Tuning 的方向進行，我們可以繼續再來看幾篇論文。
:::
