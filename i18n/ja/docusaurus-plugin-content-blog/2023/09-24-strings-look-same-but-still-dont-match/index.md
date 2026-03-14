---
slug: strings-look-same-but-still-dont-match
title: 同じに見えるのに、なぜ文字列比較は失敗するのか？
authors: Z. Yuan
date: 2023-09-24T09:56:27+08:00
tags: [unicode, python, javascript, text-processing, debugging]
image: /img/2023/0924-unicode-string-traps.svg
description: 文字列は見た目が同じでも一致しないことがあります。原因はたいてい Unicode 正規化、不可視文字、そして人間の目への過信です。
---

2 つの文字列を見る。

見た目は同じ。

`==` で比較する。

失敗する。

この手の事故が起きると、人はだいたい次の順で壊れます。

1. まず自分の目を疑う
2. 次に文字コードを疑う
3. 最後に宇宙の悪意を疑う

たいてい宇宙は無実です。

問題はもっと地味です。

> **見た目が同じことと、code point や byte 列が同じことは別です。**

この記事では、よくある落とし穴を整理します。

1. 同じ字形でも Unicode の構成が違う
2. 不可視文字が混ざる
3. 全角・半角、ダッシュ、空白の違い
4. `trim()` では足りない理由
5. いつ正規化すべきか、いつ正規化してはいけないか

例は Python と JavaScript の両方を使います。どちらもこの分野では十分に厄介です。

<!-- truncate -->

## まず原則：同じ見た目でも、同じ code point とは限らない

代表例は `é` です。

これは次の 2 通りで表現できます。

- 1 文字の `U+00E9`
- `e` と結合アクセント `U+0301`

表示上は同じです。

でも内部表現は違います。

### Python

```python
s1 = "é"
s2 = "e\u0301"

print(s1 == s2)          # False
print(len(s1), len(s2))  # 1 2
print([hex(ord(c)) for c in s1])
print([hex(ord(c)) for c in s2])
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

画面だけ見ていると理不尽ですが、コンピュータ側の挙動としては完全に正常です。

## 対処 1：Unicode 正規化で表現形式をそろえる

万能薬ではありませんが、最初にやるべきこととしてはかなり正しいです。

代表的な形式は次の通りです。

- `NFC`: 合成寄りの標準形
- `NFD`: 分解寄りの形式
- `NFKC`: 互換正規化。より強く畳み込む
- `NFKD`: 互換分解

普通の文字列比較なら、まず `NFC` を考えます。

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

### `NFKC` はいつ使うべきか

向いているのは、たとえば次のような場面です。

- ゆるい検索
- ユーザー入力の揺れを吸収したい識別子
- 全角英数字を ASCII に寄せたい場合

```python
import unicodedata

print(unicodedata.normalize("NFKC", "ＡＢＣ１２３"))
# ABC123
```

便利です。

同時に、雑に使うと危険です。

`NFKC` は単なる形式統一ではなく、互換文字をより積極的に畳み込みます。

検索には向いています。

パスワード、署名、法的原文、厳密な入力保持には向かないことがあります。

なので雑にまとめるとこうです。

- **検索・あいまい比較**: `NFKC` は有力
- **保存・セキュリティ上厳密な比較**: まずは `NFC`、あるいは原文そのものを保持

## 対処 2：不可視文字を目視ではなくコードで暴く

もう 1 つの定番事故が不可視文字です。

- zero-width space `U+200B`
- no-break space `U+00A0`
- word joiner `U+2060`
- BOM `U+FEFF`
- タブ、復帰、変な改行

これらは次の経路で平然と混ざります。

- Web ページからのコピー
- Excel / Word の出力
- IME
- OCR 後処理
- 外部 API

例：

```python
s1 = "token=abc123"
s2 = "token=abc123\u200b"

print(s1 == s2)  # False
print(repr(s2))
```

`repr()` を出さないと、そもそも異物があることに気づけないことがあります。

### デバッグ用の定番

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

上品ではありません。

でも効きます。

デバッグで大事なのは、だいたい品格ではなく再現性です。

## `trim()` は便利だが、救世主ではない

文字列の不具合を見ると、すぐにこうしたくなります。

- Python: `s.strip()`
- JavaScript: `s.trim()`

役には立ちます。

しかし十分ではありません。

理由は単純です。

1. 先頭と末尾しか触らない
2. Unicode の合成・分解問題は解決しない
3. ダッシュや空白の種類違いまでは吸収しない

## ダッシュは全部 `-` ではない

実務では次のような文字が混ざります。

- Hyphen-minus: `-` (`U+002D`)
- Non-breaking hyphen: `‑` (`U+2011`)
- En dash: `–` (`U+2013`)
- Em dash: `—` (`U+2014`)
- Minus sign: `−` (`U+2212`)

人間は全部「ダッシュっぽいもの」と読みます。

パーサはそうしません。

もし正規表現、split、ファイル名ルール、slug 規則が ASCII の `-` だけを期待しているなら、他は普通に事故要因です。

フィールドの意味として ASCII しか許さないべきなら、最初からそう制約した方が安いです。

後から「たぶんこういう意味だろう」と推測するのは、だいたい保守コストの前払いです。

## クリーニングの基本：原文を保持し、別に canonical form を作る

これはかなり重要です。

ユーザー入力を受けた瞬間に書き換えてしまうのは、長期的にはあまり賢くありません。

安定しやすい流れはこうです。

1. **原文は保持する**
2. 検索・重複判定・緩い比較用に **canonical form** を作る
3. ルールを明文化し、再現可能にする

Python の例：

```python
import unicodedata


def canonicalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00A0", " ")
    text = text.replace("\u200B", "")
    text = text.strip()
    return text
```

もっと緩い検索用なら：

```python
import re
import unicodedata


def search_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

この 2 つを混ぜると、あとで静かに面倒が育ちます。

## Python では `lower()` より `casefold()` を検討する

Unicode を含む大小無視比較では、Python では `lower()` より `casefold()` の方が適切な場面があります。

```python
print("Straße".lower())
print("Straße".casefold())
```

出力：

```text
straße
strasse
```

JavaScript にはこれと完全に同等の `casefold()` はありません。通常は：

- `toLowerCase()`
- `toLocaleLowerCase()`
- それに自前の正規化ルール

という構成になります。

なので、多言語を真面目に扱う比較ルールは、フロントで各自が気分で書くより、バックエンド側で一元化した方が安定します。

## パスワード、署名、token に雑な正規化を入れない

ここは分けて強調しておきます。

文字列事故を見ると、ついこう考えがちです。

> 「じゃあ全部 normalize すれば平和では？」

平和にはなりません。

むしろバグが見えにくくなることがあります。

次のようなデータは、ゆるい正規化を勝手にかけるべきではありません。

- パスワード
- HMAC / API 署名
- JWT / token
- ハッシュ入力
- 原文保持が必要な法務・監査系データ

こういうものに必要なのは：

- byte 単位の安定性
- 明示的なルール
- 入力内容の勝手な再解釈をしないこと

警告は出していい。

怪しい文字を検出していい。

でも、黙って「直す」のはだいたい危ないです。

## 実務で使う確認手順

「同じに見えるのに比較が失敗する」とき、私はだいたい次の順で見ます。

1. `repr()` / `JSON.stringify()` を出す
2. code point を全部並べる
3. zero-width や特殊空白の混入を疑う
4. `NFC` 後に一致するか確認する
5. フィールドの意味として `NFKC` を許せるか考える
6. ルールを共通関数に寄せる

Unicode が難しすぎるから事故る、というよりは、

- A は `trim()`
- B は `lower()`
- C は `NFKC`
- D は何もしない

みたいなチーム構成で事故ることの方が多いです。

民主的ではあります。

運用しやすくはありません。

## 実用ヘルパーを 2 つ

### Python

```python
import re
import unicodedata

ZERO_WIDTH = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\ufeff",
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

絶対解ではありません。

ただ、素の `==` をそのまま信仰するよりは、かなり現実的です。

## まとめ

文字列バグが面倒なのは、見た目には問題なさそうに見えることです。

でも、code point が違い、不可視文字が入り、正規化方針がスタック全体で揃っていないと、システムは静かに壊れます。

役に立つ原則は次の 4 つです。

1. **実際の文字を確認する**
2. **フィールドの意味に応じて洗浄強度を決める**
3. **原文を保持し、canonical form は別に作る**
4. **ルールを一箇所に集める**

コンピュータはここで意地悪をしているわけではありません。

ただ、あなたの意図を勝手に補完しないだけです。

冷たい態度です。

でも、かなり仕事はできる態度でもあります。
