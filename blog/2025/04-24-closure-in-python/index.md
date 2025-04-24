---
slug: closure-in-python
title: Closure 是什麼？
authors: Z. Yuan
image: /img/2025/0424.jpg
tags: [python, closure]
description: 簡單介紹 Closure 的概念。
---

寫程式的時候，偶爾會聽到「Closure」這個詞。

這不是什麼陌生的概念。我們常常在用，只是不一定意識到它的名字。

<!-- truncate -->

## 函式是第一類物件

在 Python 裡，函式不只是語法糖的工具，而是具備完整權能的物件。

你可以：

- 將函式指派給變數
- 將它當作參數傳遞
- 從另一個函式中將它作為結果回傳

```python
def greet(name):
    return f"Hello, {name}"

say_hello = greet
print(say_hello("Alice"))
# => Hello, Alice
```

這意味著函式可以像資料一樣被操作，也能與其他邏輯共構，形成模組化的行為單位。

## 作用域中的函式生成

Python 允許在函式內部定義其他函式，形成巢狀的結構：

```python
def outer():
    def inner():
        print("Hello from the inside")
    inner()
```

在這裡，`inner()` 僅能在 `outer()` 的作用域中存活，外部無法直接呼叫。

但我們可以換個方式撰寫，改變它的命運：

- 將函式作為結果傳遞

  ```python
  def outer():
      def inner():
          print("I’m still alive.")
      return inner

  escaped = outer()
  escaped()  # => I’m still alive.
  ```

在這段程式碼中，雖然 `outer()` 已經結束，但 `inner()` 仍可被呼叫。

原因在於 `inner()` 被「帶出來」的同時，也攜帶了它需要的執行上下文。

## Closure

終於來到主題了。

Closure 是一種語言機制，允許函式捕捉其外部作用域中的變數，並在函式結束後仍然能夠使用這些變數。

下面看一個例子：

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

在這裡，`factor` 是 `multiply()` 的自由變數：

- **它不是在 `multiply()` 的內部定義，但被用到了。**

當 `make_multiplier()` 執行結束後，這個 `factor` 並沒有消失。

它被「封裝」在 `multiply()` 裡，一起被返回。

這樣的組合，就是所謂的 **Closure**。

## 如何辨認 Closure？

可以從函式的 `__closure__` 屬性來觀察：

```python
>>> triple.__closure__
(<cell at 0x...: int object at 0x...>,)

>>> [c.cell_contents for c in triple.__closure__]
[3]
```

- `__closure__` 會列出函式中被捕捉的自由變數
- `cell_contents` 取出這些變數的實際內容

這不是什麼神秘現象，只是一種語言機制的自然結果。

## 常見用途與場景

- **函式工廠**：根據輸入參數產出具狀態的自訂函式
- **計數器／快取**：保留有限狀態，避免額外開 `class`
- **裝飾器 (`@decorator`)**：常見的實作方式就是基於 Closure 結構疊加
- **依賴注入**：將資料隱性地綁定，避免污染全域狀態

當你需要 **保存少量狀態**、又不想動用完整物件導向設計時，Closure 是一個恰到好處的工具。

## 小結

Closure 不難懂，它的本質只是：

1. **捕捉**：將自由變數的值保存下來
2. **打包**：與函式本體一同封裝
3. **延續**：即使原始作用域不再，仍能正常運作

當你遇到 `__closure__`，不需要驚訝。它只是當時環境的一個封存版本，保留了那一刻的資料狀態。

這些值像是程式的記憶片段，陪著函式一起前行。
