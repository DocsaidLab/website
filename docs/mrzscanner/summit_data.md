---
sidebar_position: 9
---

# 資料集提交

真實世界千奇百怪，你一定會遇到不合用的時候。

我們的模型也是如此，可能無法應對所有情況。

如果你在使用的過程中，發現某些情況下我們的模型無法正確處理，我們建議你提供一些資料集給我們，我們會根據你提供的資料集，進行模型的調整和優化。

我們非常感謝你願意提供資料集，並且會在第一時間進行測試和整合。

## 格式說明

在這個任務中，你需要提供的是：

- **帶有文字標籤的 MRZ 影像，就像我們之前提到的 MIDV-2020 資料集一樣。**

---

例如：有一張影像，上面有 MRZ 區域，標籤資料大概會長這樣：

```json
{
  "img_01": {
    "img_path": "path/to/image.jpg",
    "mrz_1": "P<USALAST<<FIRST<MIDDLE<NAME<<<<<<<<<<<<<<<<",
    "mrz_2": "1234567890USA1234567890<<<<<<<<<<<<<<<<<<<<4"
  },
  "img_02": {
    "img_path": "path/to/image.jpg",
    "mrz_1": "P<USALAST<<FIRST<MIDDLE<NAME<<<<<<<<<<<<<<<<",
    "mrz_2": "1234567890USA1234567890<<<<<<<<<<<<<<<<<<<<4"
  }
}
```

---

我們建議你將資料上傳至你的 google 雲端，並透過 [**電子郵件**](#聯絡我們) 提供給我們連結，我們會在收到你的資料後，盡快進行測試和整合。若你所提供的資料不符合我們的需求，我們會在第一時間通知你。

## 常見問題

1. **我提交了的資料，效果就會好嗎？**

   - 不好說。雖然我們讓模型看過你所提供的資料，但這不表示這份資料的特徵能夠對模型產生足夠的影響力。只能說有看過比沒看過好，但是不見得就會有很大的提升。

2. **檔案名稱的重要性如何？**

   - 檔案名稱不是主要關注點，只要能正確連接到相應的影像即可。

3. **影像格式有何建議？**
   - 建議使用 jpg 格式以節省空間。

---

## 聯絡我們

若需要更多幫助，請透過電子郵件與我們聯繫：**docsaidlab@gmail.com**
