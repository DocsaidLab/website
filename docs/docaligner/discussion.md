---
sidebar_position: 7
---

# 結果與討論

綜合之前的實驗，我們得到了一個效果不錯的模型。

這裡我們將討論一些我們在訓練過程中的一些心得和經驗。

---

- 雖然我們的模型可以達到接近 SOTA 的分數，但現實場景遠比這個資料集複雜，因此不用過於在意這個分數，我們只是想要證明我們的模型是有效的。

- 在實驗中，我們發現目前設計的模型架構對於 Zero-shot 的能力並不好，也就是說，模型對於新的場景，需要進行微調才能達到最佳效果。在未來我們應該要更深入地探索更具有泛化能力的模型架構。

- 如同模型設計的章節中提到，我們沒有辦法直接解決放大誤差的挑戰，因此使用「熱圖回歸模型」的穩定性遠高於「點回歸模型」。

- 我們預設使用 `FastViT_SA24` 作為熱圖模型的 Backbone，因為它的效果和運算量都很好。

- 經過實驗，`BiFPN`（3 層） 效果優於 `FPN`（6 層），因此我們推薦你使用 `BiFPN` 作為 Neck 部分的配置。但是在我們實作的 `BiFPN` 中有用到 `einsum` 的操作，可能會導致其他推論框架的困擾，因此若你在使用 `BiFPN` 時候遇到轉換上的錯誤，可以考慮改為 `FPN` 模型。

- 儘管「熱圖回歸模型」表現穩定，但由於需要在高解析度的特徵圖上進行監督，因此模型的運算量遠高於「點回歸模型」。

- 但我們仍無法割捨「點回歸模型」的優點，包含但不限於：可以預測圖面範圍之外的角點；計算量低及快速簡單的後處理流程等。因此我們會持續探索和優化「點回歸模型」，以提升其效果。
