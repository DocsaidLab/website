---
slug: why-identical-strings-still-fail
title: 看起來一樣，為什麼字串還是比對失敗？
authors: Z. Yuan
image: /img/2026/0314-string-compare-unicode.svg
tags: [unicode, python, text-processing]
description: 字串看起來一樣，不代表它們真的一樣。問題通常出在 Unicode、不可見字元，以及你對電腦的過度信任。
---

你一定遇過這種事：

兩段字看起來一模一樣，結果程式就是比對失敗。

然後你盯著螢幕五分鐘，開始懷疑自己是不是瞎了。

通常不是你瞎了。

是電腦太誠實。

<!-- truncate -->

對人類來說，兩個字「看起來一樣」，大多就會自動被腦補成同一件事。

對程式不是。

程式不看感覺，它看的是：

- code point
- byte sequence
- 正規化形式
- 有沒有混進不可見字元

只要其中一項不同，它就很有可能判定：

> **不一樣就是不一樣。**

很冷酷，但也沒毛病。

## 一個最經典的例子：`é`

先看這兩段字：

```python
s1 = "café"
s2 = "cafe\u0301"

print(s1 == s2)
```

很多人直覺會以為輸出是 `True`。

實際上通常是：

```python
False
```

因為這兩個 `é`，雖然長得一樣，但底層不是同一種表示法：

- `é`：單一 code point
- `e` + `◌́`：字母 `e` 加上 combining acute accent

畫面看起來差不多。

但對字串比較來說，它們不是同一串東西。

## 為什麼會這樣？

因為 Unicode 並不是「一個字長怎樣」那麼簡單。

它更像是一套規則，告訴你：

- 字元怎麼編號
- 字元怎麼組合
- 不同平台怎麼表示它們

這裡有三個層次要分清楚。

### 1. Code point

Unicode 會替每個字元指派一個編號，例如：

- `A` → `U+0041`
- `é` → `U+00E9`

這是最基本的身分證號。

### 2. Grapheme

使用者眼中看到的一個「字」，不一定只由一個 code point 組成。

像剛才的 `e` + 重音符號，就是一個很典型的例子。

人類看到的是一個字。

程式看到的可能是兩個成分。

### 3. Encoding

等到字串真的要存成 bytes 時，又會有 UTF-8、UTF-16 之類的編碼問題。

所以「看起來一樣」這件事，在不同層次上都可能失手。

## 常見地雷，不只重音符號

這類問題不只發生在法文或特殊字元，很多平常資料都會踩到。

### 一、全形與半形

```python
s1 = "ABC123"
s2 = "ＡＢＣ１２３"

print(s1 == s2)  # False
```

對人類來說，這只是字比較胖。

對程式來說，是完全不同的字元。

### 二、不可見字元

最討厭的通常不是長得不一樣的字，而是你看不到的字。

例如：

- zero-width space
- non-breaking space
- directional marks
- 文字從網頁複製時帶進來的控制字元

這些東西混進資料後，畫面還是很乾淨，只有你的比對結果開始發瘋。

### 三、大小寫不是你想的那麼簡單

很多人以為 case-insensitive compare 只要 `lower()` 就好。

不一定。

某些語言的大小寫轉換規則沒那麼樸素，Unicode 也不是全世界都只講英文。

如果你真的要做 Unicode 層級的大小寫無關比較，通常會更偏向使用：

```python
text.casefold()
```

而不是只靠 `lower()`。

## 解法：先正規化，再談比較

這種問題的標準處理方式叫做 **Unicode normalization**。

Python 內建的 `unicodedata` 就能做：

```python
import unicodedata

s1 = "café"
s2 = "cafe\u0301"

n1 = unicodedata.normalize("NFC", s1)
n2 = unicodedata.normalize("NFC", s2)

print(n1 == n2)  # True
```

這時候兩邊就會先被整理成相同的表示形式，再做比較。

終於肯講人話了。

## NFC、NFD、NFKC、NFKD 到底差在哪？

這四個名字第一次看很像亂碼，實際上只是在回答兩個問題：

1. 要不要拆開？
2. 要不要做相容性轉換？

### 1. NFC

**Canonical Composition**

傾向把可合併的字元組合回去。

例如：

- `e` + accent → `é`

這通常是**最保守也最常用**的選擇。

如果你的需求是：

- 儲存一般文字
- 做穩定比對
- 保留原始語意

那大多數情況下，先試 `NFC` 就對了。

### 2. NFD

**Canonical Decomposition**

把可組合字元拆開。

比較常見於某些文字分析流程，或你真的需要逐個組件處理字元時。

一般業務系統不太會把它當預設格式。

### 3. NFKC

**Compatibility Composition**

除了標準正規化之外，還會做「相容性」層級的轉換。

例如某些：

- 全形字
- 相容字元
- 視覺上接近但語意被 Unicode 視為可折疊的形式

都可能被收斂成更統一的結果。

這很有用。

也很危險。

因為它做得比較多，所以適合：

- 搜尋索引
- 使用者輸入清理
- 帳號、識別碼這種你想盡量收斂格式的欄位

但如果你處理的是：

- 法律文本
- 排版敏感內容
- 必須保留原貌的資料

那就不要隨手上 `NFKC`。

### 4. NFKD

拆開版的 compatibility normalization。

除非你真的知道自己在做什麼，不然大部分時候不會先選它。

## 一個比較像樣的清理流程

實務上，比對文字通常不只做 normalization。

還會一起處理：

- Unicode normalization
- case folding
- 空白整理
- 不可見控制字元移除

例如：

```python
import re
import unicodedata


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


s1 = " Docsaid\u00A0Lab "
s2 = "docsaid lab"

print(normalize_text(s1) == normalize_text(s2))  # True
```

這個版本已經比單純 `strip().lower()` 靠譜很多。

至少不會一邊自信，一邊出錯。

## 但不要什麼都正規化

這裡有個很常見的過度工程：

> 「反正 normalization 很好用，那我全部欄位都先做一遍。」

別。

有些資料不能亂動。

例如：

- 密碼
- token
- 簽章資料
- 雜湊前原文
- 需要逐 byte 保真的欄位

這些東西只要你先正規化，後面就可能整串對不起來。

有些系統甚至不是壞在比較，而是壞在你「好心幫它整理過」。

工程界有很多 bug，就是這樣被做出來的。

## 什麼時候該用哪一種？

如果你懶得記規格，可以先記這個粗暴版本：

- **一般文字儲存 / 顯示**：先考慮 `NFC`
- **搜尋、帳號、使用者輸入比對**：考慮 `NFKC` + `casefold()`
- **安全敏感資料**：不要亂正規化
- **看到明明一樣卻比對失敗**：先懷疑 Unicode，再懷疑人生

這個順序比較省時間。

## 怎麼快速排查？

當你懷疑字串有鬼，不要只 `print(text)`。

那通常沒有用。

請直接看它的表示方式：

```python
text = "cafe\u0301"

print(repr(text))
print([hex(ord(ch)) for ch in text])
```

輸出會像這樣：

```python
'cafe\u0301'
['0x63', '0x61', '0x66', '0x65', '0x301']
```

這時你就知道，不是資料庫在針對你，也不是 Python 今天心情不好。

是字串裡真的多了一個 combining mark。

## 最後

字串比對失敗，很多時候不是邏輯太複雜。

而是你以為「看起來一樣」就等於「底層一樣」。

這個假設對人類合理，對電腦不合理。

電腦不會幫你腦補。

它只會安靜地回你一個 `False`，然後看你自己崩潰。

所以，如果你有以下症狀：

- 從網頁貼過來的字一直對不起來
- 使用者名稱明明一樣卻查不到
- 多語系文本在搜尋和去重時怪怪的
- 比對前你只做了 `lower().strip()` 然後很有信心

那你現在該做的事情大概不是再加一個 `if`。

而是先去把 Unicode 正規化補上。

這比較像在修 bug，不像在祈禱。

## 參考資料

- [Unicode Standard Annex #15: Unicode Normalization Forms](https://unicode.org/reports/tr15/)
- [Python `unicodedata` Documentation](https://docs.python.org/3/library/unicodedata.html)
- [Python `str.casefold`](https://docs.python.org/3/library/stdtypes.html#str.casefold)
