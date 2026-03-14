---
slug: why-identical-strings-still-fail
title: 同じに見えるのに、なぜ文字列比較は失敗するのか？
authors: Z. Yuan
image: /ja/img/2026/0314-string-compare-unicode.svg
tags: [unicode, python, text-processing]
description: 見た目が同じでも、文字列が同じとは限りません。原因はたいてい Unicode、不可視文字、そしてコンピュータへの雑な期待です。
---

こういう経験はたぶん一度はあるはずです。

文字列はまったく同じに見えるのに、比較すると失敗する。

そして画面を五分くらい見つめたあと、自分の目がおかしくなったのかと思い始める。

たいてい、目は悪くありません。

コンピュータが妙に正直なだけです。

<!-- truncate -->

人間にとって「同じに見える」は、だいたい十分です。

コードにとっては違います。

コードが見ているのは、感覚ではなく次のようなものです。

- code point
- byte sequence
- 正規化形式
- 文字列の中に不可視文字が混じっていないか

そのどれかが違えば、答えはこうなります。

> **違うものは違う。**

冷たいですが、別に間違ってはいません。

## 定番の例：`é`

まずはこの二つを見てください。

```python
s1 = "café"
s2 = "cafe\u0301"

print(s1 == s2)
```

多くの人は `True` を期待します。

でも実際には、たいていこうなります。

```python
False
```

理由は単純で、この二つの `é` は内部表現が同じではないからです。

- `é`：単一の code point
- `e` + `◌́`：`e` のあとに combining acute accent

画面では同じに見えても、文字列としては別物です。

## どうしてこうなるのか？

Unicode は、単なる「この字はこう見える」という話ではありません。

むしろ次のことを定める仕組みです。

- 文字にどう番号を振るか
- 文字をどう組み合わせるか
- それを各環境でどう表現するか

ここでは三つの層を分けて考えると分かりやすいです。

### 1. Code point

Unicode は各文字に識別子を割り当てます。たとえば：

- `A` → `U+0041`
- `é` → `U+00E9`

文字の身分証みたいなものです。

### 2. Grapheme

ユーザーが「一文字」と認識するものが、必ずしも一つの code point とは限りません。

`e` とアクセントの組み合わせは、その典型です。

人間は一文字と見ます。

プログラムは二つの部品と見ているかもしれません。

### 3. Encoding

さらに文字列が bytes になる段階では、UTF-8 や UTF-16 のような encoding も関わってきます。

つまり「同じに見える」は、いくつもの層で簡単に裏切られます。

## よくある地雷は、アクセントだけではない

この問題はフランス語や特殊文字だけの話ではありません。普段のデータでも普通に起きます。

### 1. 全角と半角

```python
s1 = "ABC123"
s2 = "ＡＢＣ１２３"

print(s1 == s2)  # False
```

人間から見ると、少し幅が広いだけです。

プログラムから見ると、別の文字です。

### 2. 不可視文字

面倒なのは、違って見える文字より、見えない文字です。

たとえば：

- zero-width space
- non-breaking space
- directional marks
- Web ページや Office 文書から混入した制御文字

これらが入っても画面はきれいなままです。

壊れるのは、だいたい比較処理の方です。

### 3. 大文字・小文字も思ったほど単純ではない

case-insensitive compare は `lower()` で十分、と思っている人は多いです。

残念ながら、必ずしもそうではありません。

Unicode は英語だけの世界ではありませんし、言語によっては大小文字変換がもっと癖のある動きをします。

Unicode を意識した大小文字無視の比較なら、たいていはこちらの方がましです。

```python
text.casefold()
```

`lower()` だけで済ませない方が安全です。

## 解決策：まず正規化してから比較する

こういう問題の標準的な対処は **Unicode normalization** です。

Python なら `unicodedata` が最初から使えます。

```python
import unicodedata

s1 = "café"
s2 = "cafe\u0301"

n1 = unicodedata.normalize("NFC", s1)
n2 = unicodedata.normalize("NFC", s2)

print(n1 == n2)  # True
```

両方を同じ正規化形式にそろえてから比較すれば、ようやく話が通じます。

## NFC、NFD、NFKC、NFKD は何が違うのか？

最初は暗号みたいに見えますが、実際には二つの問いに答えているだけです。

1. 分解するか？
2. compatibility 変換までやるか？

### 1. NFC

**Canonical Composition**

可能なものは合成した形に寄せます。

たとえば：

- `e` + accent → `é`

これは一番無難で、よく使われる選択です。

用途が次のようなものなら、まず `NFC` を考えれば大きく外しません。

- 普通の文字列保存
- 安定した比較
- 意味を保ったまま整形したい場合

### 2. NFD

**Canonical Decomposition**

合成文字を分解します。

文字の構成要素を個別に扱いたい処理では役立ちますが、一般的な業務システムの保存形式としてはあまり選ばれません。

### 3. NFKC

**Compatibility Composition**

標準的な正規化に加えて、compatibility レベルの変換も行います。

つまり、たとえば次のようなものが、より統一された形に寄せられる可能性があります。

- 全角文字
- compatibility 文字
- 見た目が似ていて Unicode 上は折りたためる形式

便利です。

同時に、雑に使うと危険です。

向いているのは：

- 検索インデックス
- ユーザー入力の整理
- ユーザー名や識別子の比較

逆に、次のようなものには慎重になるべきです。

- 法的に厳密な文面
- レイアウト依存の内容
- 元の見た目を正確に残す必要があるデータ

### 4. NFKD

compatibility normalization の分解版です。

明確な理由がないなら、最初に選ぶことはあまりありません。

## 実務では、もう少しまとめて処理する

実際の文字列比較は、normalization だけで終わらないことが多いです。

たとえば次のような処理も一緒に入ります。

- Unicode normalization
- case folding
- 空白整理
- 不可視の整形文字の除去

例を挙げると、こうなります。

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

これなら `strip().lower()` よりはずっとまともです。

少なくとも、自信満々に間違える確率は下がります。

## ただし、何でも正規化すればよいわけではない

ここでありがちな過剰対応があります。

> 「正規化は便利だ。全部のフィールドにかけよう。」

やめた方がいいです。

触ってはいけないデータがあります。

たとえば：

- パスワード
- token
- 署名対象データ
- hash 前の原文
- byte 単位で厳密性が必要なフィールド

こういうものを勝手に正規化すると、後で全部つじつまが合わなくなります。

比較が難しいのではなく、誰かが親切のつもりで壊しているだけ、という事故は珍しくありません。

## では、どれを使うべきか？

仕様を全部覚えたくないなら、この雑だが実用的なルールで十分です。

- **一般的な文字列保存 / 表示**：まず `NFC`
- **検索、ユーザー名、入力比較**：`NFKC` + `casefold()` を検討
- **セキュリティ敏感なデータ**：むやみに正規化しない
- **見た目は同じなのに比較が失敗する**：まず Unicode を疑う

この順番の方が時間を無駄にしません。

## どうやって素早く調べるか？

文字列に何か潜んでいそうなら、`print(text)` だけでは足りません。

たいてい、それでは何も分かりません。

表現を直接見ます。

```python
text = "cafe\u0301"

print(repr(text))
print([hex(ord(ch)) for ch in text])
```

こんな出力になります。

```python
'cafe\u0301'
['0x63', '0x61', '0x66', '0x65', '0x301']
```

これで、データベースがあなたを嫌っているわけでも、Python が急に気難しくなったわけでもないと分かります。

文字列の中に、本当に combining mark が入っているだけです。

## 最後に

文字列比較の失敗は、必ずしもロジックが難しいせいではありません。

多くの場合は、「見た目が同じなら中身も同じだろう」という前提が崩れているだけです。

その前提は人間には自然です。

コンピュータには自然ではありません。

コンピュータは補完してくれません。

静かに `False` を返して、こちらに学習を要求してくるだけです。

もし今あなたが、こんな症状を見ているなら：

- Web から貼った文字列がどうしても一致しない
- 同じに見えるユーザー名が検索で出てこない
- 多言語テキストの検索や重複排除が妙に怪しい
- 比較前に `lower().strip()` だけやって安心していた

次にやるべきことは、`if` を足すことではたぶんありません。

Unicode 正規化です。

その方が、祈るよりはずっと工学的です。

## 参考資料

- [Unicode Standard Annex #15: Unicode Normalization Forms](https://unicode.org/reports/tr15/)
- [Python `unicodedata` Documentation](https://docs.python.org/3/library/unicodedata.html)
- [Python `str.casefold`](https://docs.python.org/3/library/stdtypes.html#str.casefold)
