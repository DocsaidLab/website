---
slug: closure-in-python
title: Closure とは？
authors: Z. Yuan
image: /ja/img/2025/0424.jpg
tags: [python, closure]
description: Closure の概念を簡単に紹介します。
---

プログラムを書くとき、「Closure」という言葉を耳にすることがあります。

これは珍しい概念ではありません。私たちはよく使っていますが、その名前に気づいていないだけかもしれません。

<!-- truncate -->

## 関数は第一級オブジェクト

Python では、関数は単なる構文糖ではなく、完全な権限を持つオブジェクトです。

あなたは次のことができます：

- 関数を変数に割り当てる
- 関数を引数として渡す
- 他の関数から結果として関数を返す

```python
def greet(name):
    return f"Hello, {name}"

say_hello = greet
print(say_hello("Alice"))
# => Hello, Alice
```

これは、関数がデータのように操作でき、他のロジックと組み合わせてモジュール化された振る舞い単位を形成できることを意味します。

## スコープ内で関数を生成

Python では、関数内に他の関数を定義することができ、ネストされた構造を形成します：

```python
def outer():
    def inner():
        print("Hello from the inside")
    inner()
```

ここでは、`inner()` は `outer()` のスコープ内でのみ生きており、外部から直接呼び出すことはできません。

しかし、異なる方法で書き直すと、その運命を変えることができます：

- 関数を結果として返す

  ```python
  def outer():
      def inner():
          print("I’m still alive.")
      return inner

  escaped = outer()
  escaped()  # => I’m still alive.
  ```

このコードでは、`outer()` が終了しても、`inner()` は呼び出し可能です。

その理由は、`inner()` が「持ち出される」際に、その実行コンテキストも一緒に持ち出されているからです。

## Closure

さて、ついに本題に入ります。

Closure とは、関数がその外部スコープにある変数をキャプチャし、関数が終了した後でもその変数を使用できるようにする言語機構です。

以下の例を見てみましょう：

```python
def make_multiplier(factor):
    def multiply(n):
        return n * factor
    return multiply

triple = make_multiplier(3)
double = make_multiplier(2)

print(triple(10))  # 30
print(double(10))  # 20
```

ここでは、`factor` が `multiply()` の自由変数です：

- **`multiply()` 内で定義されていませんが、使用されています。**

`make_multiplier()` が終了した後でも、この `factor` は消えません。

それは `multiply()` 内に「封じ込められ」、一緒に返されます。

このような組み合わせが、いわゆる **Closure** です。

## Closure の見分け方

関数の `__closure__` 属性から確認できます：

```python
>>> triple.__closure__
(<cell at 0x...: int object at 0x...>,)

>>> [c.cell_contents for c in triple.__closure__]
[3]
```

- `__closure__` は関数内でキャプチャされた自由変数をリストアップします
- `cell_contents` でこれらの変数の実際の内容を取得できます

これは何か神秘的な現象ではなく、単に言語機構の自然な結果です。

## 一般的な使用例とシーン

- **関数工場**：入力パラメータに基づいて状態を持つカスタム関数を生成
- **カウンタ／キャッシュ**：有限の状態を保持し、追加で `class` を使わない
- **デコレーター (`@decorator`)**：一般的な実装方法は Closure 構造に基づいて積み重ねる
- **依存注入**：データを暗黙的にバインドし、グローバル状態の汚染を避ける

少量の状態を **保存したい** が、完全なオブジェクト指向設計を使いたくない場合、Closure は適切なツールです。

## まとめ

Closure は難しくありません。その本質は以下の通りです：

1. **キャプチャ**：自由変数の値を保存する
2. **パッケージング**：関数本体と一緒に封装する
3. **継続**：元のスコープがなくなっても、正常に動作し続ける

`__closure__` を見かけても驚かないでください。それはその時の環境を封存したバージョンであり、その瞬間のデータ状態を保持しています。

これらの値は、プログラムの記憶の断片のようなもので、関数とともに前進します。
