---
slug: strings-look-same-but-still-dont-match
title: 看起來一樣，為什麼字串還是比對失敗？
authors: Z. Yuan
date: 2023-09-24T09:56:27+08:00
tags: [unicode, python, javascript, text-processing, debugging]
image: /img/2023/0924-unicode-string-traps.svg
description: 字串看起來一樣，不代表它們真的一樣。問題通常出在 Unicode、不可見字元、正規化，以及你對電腦的過度信任。
---

你看到兩個字串。

它們看起來一樣。

你用 `==` 一比。

失敗。

這時候人通常會進入三個階段：

1. 先懷疑自己眼花
2. 再懷疑編碼壞掉
3. 最後開始懷疑整個宇宙

其實大多數情況下，宇宙沒有針對你。

只是字串這種東西，**長得像**跟**真的是同一串位元**，本來就是兩回事。

這篇想拆幾個最常見的坑：

1. Unicode 組成不同，但畫面一樣
2. 混進不可見字元
3. 全形半形、不同 dash、不同空白
4. 你以為 trim 過就沒事，其實沒有
5. 該在什麼時候正規化，什麼時候不要亂正規化

我會用 Python 和 JavaScript 都示範一次，因為這兩邊都很會坑人，只是坑法略有地方特色。

<!-- truncate -->

## 先講結論：字串一樣，不等於 code point 一樣

如果你現在正卡在：

- 資料庫查不到
- API 簽名對不上
- 搜尋結果怪怪的
- 使用者說「我明明貼的一樣」

那先不要急著 blame encoding。

先記住這句：

> **你看到的是字形，電腦比的是位元序列或 code point 序列。**

兩者常常不是同一回事。

最經典的例子是字母 `é`。

它可以是：

- 一個單一字元：`U+00E9`
- 也可以是：`e` + 結合重音 `U+0301`

畫面上都像 `é`。

但底層不一樣。

### Python

```python
s1 = "é"
s2 = "e\u0301"

print(s1 == s2)          # False
print(len(s1), len(s2))  # 1 2
print([hex(ord(c)) for c in s1])
print([hex(ord(c)) for c in s2])
```

輸出大概會像這樣：

```text
False
1 2
['0xe9']
['0x65', '0x301']
```

### JavaScript

```js
const s1 = "é";
const s2 = "e\u0301";

console.log(s1 === s2); // false
console.log(s1.length, s2.length); // 1 2
console.log([...s1].map(ch => ch.codePointAt(0).toString(16)));
console.log([...s2].map(ch => ch.codePointAt(0).toString(16)));
```

如果你只看畫面，你會覺得這根本同一個字。

電腦不這麼想。

它比較冷酷，也比較誠實。

## 解法一：用 Unicode normalization 先把字串整理成同一種形式

這不是萬靈丹，但通常是第一步。

常見形式有：

- `NFC`: 偏向組合後的標準形式
- `NFD`: 偏向拆解後的形式
- `NFKC`: 相容性正規化，會做更激進的折疊
- `NFKD`: 相容性拆解版本

大多數一般文字比對，先考慮 `NFC`。

### Python

```python
import unicodedata

s1 = "é"
s2 = "e\u0301"

n1 = unicodedata.normalize("NFC", s1)
n2 = unicodedata.normalize("NFC", s2)

print(n1 == n2)  # True
```

### JavaScript

```js
const s1 = "é";
const s2 = "e\u0301";

console.log(s1.normalize("NFC") === s2.normalize("NFC")); // true
```

### 什麼時候用 `NFKC`？

當你做的是：

- 使用者輸入的寬鬆搜尋
- 帳號、代號、標籤這類想收斂輸入形式的欄位
- 想把全形英數折成半形英數

例如：

```python
import unicodedata

print(unicodedata.normalize("NFKC", "ＡＢＣ１２３"))
# ABC123
```

這很方便。

也很危險。

因為 `NFKC` 不只是整理，還會做**相容性折疊**。

也就是說，它有時不是「保留原文但換個標準形式」，而是「我幫你把看起來差不多的東西直接壓成同一類」。

對搜尋很有用。

對密碼、簽名、法務文本、原文保存，很可能是災難。

所以規則很簡單：

- **搜尋 / 寬鬆比對**：可以考慮 `NFKC`
- **資料保存 / 安全敏感比對**：通常只做 `NFC`，甚至保留原文另存

## 解法二：把不可見字元抓出來，不要靠肉眼 debug

另一種常見翻車點是：

- 零寬空白 `U+200B`
- 不換行空白 `U+00A0`
- word joiner `U+2060`
- BOM `U+FEFF`
- tab、carriage return、奇怪換行

這些字元很喜歡混進：

- 從網頁複製的文字
- Excel / Word 匯出的內容
- IME 輸入結果
- OCR 後處理文本
- 外部 API 回傳資料

例如：

```python
s1 = "token=abc123"
s2 = "token=abc123\u200b"

print(s1 == s2)  # False
print(repr(s2))
```

輸出：

```text
False
'token=abc123\\u200b'
```

如果你不用 `repr()`，你甚至很難發現那個字元存在。

### 我常用的 debug 方式

#### Python

```python
def inspect_string(s: str):
    for i, ch in enumerate(s):
        print(i, hex(ord(ch)), repr(ch))
```

#### JavaScript

```js
function inspectString(s) {
  [...s].forEach((ch, i) => {
    console.log(i, "U+" + ch.codePointAt(0).toString(16).toUpperCase(), JSON.stringify(ch));
  });
}
```

這種做法很土。

但有效。

debug 時，我寧可土，也不要高雅地浪費兩小時。

## `trim()` 很有用，但不要把它當神

很多人一看到字串問題就先：

- Python：`s.strip()`
- JavaScript：`s.trim()`

這可以解掉一部分問題。

但不是全部。

因為：

1. 它只處理頭尾，不處理中間
2. 對某些 Unicode 格式字元未必有你期待的效果
3. 它不會替你處理 composed/decomposed 的問題

例如這種：

```text
Hello\u0000World
Hello\u000bWorld
Hello\rWorld
Hello\nWorld
```

或：

```text
2025-09-01
2025‑09‑01
2025–09–01
2025—09—01
```

你眼裡都是 dash。

電腦眼裡不是。

## 長得像 dash，不代表就是 `-`

實務上很常出現這些：

- Hyphen-minus: `-` (`U+002D`)
- Non-breaking hyphen: `‑` (`U+2011`)
- En dash: `–` (`U+2013`)
- Em dash: `—` (`U+2014`)
- Minus sign: `−` (`U+2212`)

如果你的 parser、正則、split、檔名規則只接受 ASCII `-`，那這些都會讓你翻車。

### Python 範例

```python
samples = ["2025-09-01", "2025‑09‑01", "2025–09–01", "2025—09—01"]

for s in samples:
    print(s, [hex(ord(c)) for c in s if not c.isdigit()])
```

### 實務做法

如果欄位本質上就只該接受 ASCII，例如：

- slug
- 檔名規格
- internal ID
- command option

那就不要裝寬容。

**明確限制輸入集合**，通常比事後猜測字元意圖穩很多。

## 資料清洗的正確姿勢：保留原文，再做 canonical form

這是我比較推薦的做法。

不要一進系統就把使用者原文亂折。

比較穩的流程通常是：

1. **保留原始輸入**
2. 建立一個 **canonical form** 供搜尋 / 去重 / 比對使用
3. 規則寫死，並且可重現

例如 Python：

```python
import unicodedata


def canonicalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00A0", " ")      # nbsp -> normal space
    text = text.replace("\u200B", "")       # zero width space -> remove
    text = text.strip()
    return text
```

如果你需要更寬鬆搜尋：

```python
import re
import unicodedata


def search_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

這兩個函式不該混為一談。

- `canonicalize()`：偏保守
- `search_key()`：偏搜尋導向

把兩者混在一起，後面通常會補 bug 補到心情不太穩定。

## `lower()` 不夠，文字比對通常該考慮 `casefold()`

如果你在做不分大小寫的 Unicode 文字比對，Python 裡通常 `casefold()` 比 `lower()` 更合適。

```python
print("Straße".lower())
print("Straße".casefold())
```

輸出：

```text
straße
strasse
```

這在某些歐洲語系場景尤其重要。

JavaScript 沒有直接對等的 `casefold()`，通常只能靠：

- `toLowerCase()` / `toLocaleLowerCase()`
- 再搭配你自己的正規化規則

也就是說，如果你做的是跨語系的嚴肅全文檢索，前端順手比一比可以，真正的 canonicalization 最好放在後端做。

## 千萬別對密碼、簽名、token 亂做 normalization

這一點值得單獨拉出來講。

有些工程師一看到字串問題，就會想：

> 「那我把所有輸入都 normalize 一下，不就天下太平？」

不。

那通常只是把 bug 從「顯性」變成「更難查」。

以下資料通常**不能亂做寬鬆正規化**：

- 密碼
- HMAC / API signature
- JWT / token
- 雜湊輸入
- 法律或審計要求保真原文的欄位

這些欄位要的是：

- 明確位元一致
- 規則穩定
- 不偷偷替使用者解讀

你可以在 UI 顯示提醒。

你可以在輸入時檢測可疑字元。

但不要擅自幫它「修正」。

## 一套比較實用的排查順序

如果你遇到「看起來一樣但比對失敗」，我通常這樣查：

1. **先印 `repr()` / `JSON.stringify()`**
2. **列出每個 code point**
3. **檢查是否混入零寬或特殊空白**
4. **對照 `NFC` 後結果是否一致**
5. **確認欄位語意是否允許更激進的 `NFKC`**
6. **把規則收斂成一個共用函式，不要每個地方各自亂洗**

很多 bug 不是因為 Unicode 太複雜。

而是因為團隊裡：

- A 用 `trim()`
- B 用 `lower()`
- C 用 `NFKC`
- D 什麼都不做

最後大家都說自己是對的。

技術上來說，這很民主。

系統上來說，這很難維運。

## Python / JavaScript 各給一個實用版本

### Python

```python
import re
import unicodedata

ZERO_WIDTH = {
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\ufeff",  # BOM / zero width no-break space
}


def clean_for_search(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch not in ZERO_WIDTH)
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

### JavaScript

```js
function cleanForSearch(text) {
  return text
    .normalize("NFKC")
    .replace(/[\u200B\u200C\u200D\uFEFF]/g, "")
    .toLocaleLowerCase("en-US")
    .replace(/\s+/g, " ")
    .trim();
}
```

這不是宇宙真理。

但對「搜尋、標籤、一般使用者輸入比對」這類場景，通常已經比裸用 `==` 靠譜很多。

## 小結

字串問題最煩的地方，在於它常常**看起來像資料沒問題**。

但只要底層 code point 不同、混入不可見字元、或 normalization 策略不一致，系統就會開始表演。

所以真正有用的原則不是：

- 「看到怪字就 trim 一下」
- 「全都 lower 一下」
- 「全部丟進 NFKC」

而是：

1. **先看清楚底層字元是什麼**
2. **依欄位語意決定清洗強度**
3. **保留原文，另外建立 canonical form**
4. **把規則集中管理，不要每段程式各自發揮**

畢竟，電腦其實沒有那麼難搞。

它只是拒絕替你腦補。

這點雖然冷酷，但老實說，挺專業的。
