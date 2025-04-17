---
slug: react-hook-vs-python
title: React 到底在 Hook 什麼？
authors: Z. Yuan
image: /img/2025/0417.jpg
tags: [react, hook]
description: 拆解 React Hook 背後的概念。
---

幾年前剛開始學 React 的時候，總覺得 React 彷彿活在平行時空，難以理解。

你看看這個 Hook，也太反人類的吧？

<!-- truncate -->

## 從 Python 開始

你可能也和我一樣，都比較擅長用 Python，所以我們先從 Python 開始吧！

先來看一下最簡單的計數器，在 React 中，我們會這樣寫：

```jsx
import { useState } from "react";

function Counter() {
  const [count, setCount] = useState(0);

  function handleClick() {
    setCount(count + 1);
  }

  return <button onClick={handleClick}>Clicked! Count = {count}</button>;
}

export default Counter;
```

如果讓你用 Python 來寫計數器，你會怎麼寫？

你說很簡單？當然是用 OOP 寫啊！

## 傳統 OOP 寫法

我寫起來是這樣的，這題本身沒什麼難度：

```python
class Counter:

    def __init__(self):
        self.count = 0

    def handle_click(self):
        self.count += 1
        self.render()

    def render(self):
        print(f"Clicked! Count = {self.count}")


# 使用
counter = Counter()
counter.handle_click()  # 點一下
counter.handle_click()  # 再點一下
```

但這個寫法，和上面的 React 難以對照，所以我們接著換個寫法：

## 函數式 + 狀態綁定

不使用類別，我們直接定義一個函數，把「狀態」和「渲染邏輯」拆開，用純函數＋閉包管理狀態：

```python
def use_state(initial_value):

    state = {"value": initial_value}

    def get_value():
        return state["value"]

    def set_value(new_value):
        state["value"] = new_value

    return get_value, set_value
```

到這邊，有沒有覺得有點像了？

接著，我們建立一個狀態：

```python
get_count, set_count = use_state(0)
```

這已經是純純的 Python，別再說我在寫 React 了。

取得 `get_count` 和 `set_count` 兩個函數，然後我們來寫一個「點選事件」函數：

```python
def render():
    print(f"Clicked! Count = {get_count()}")

def handle_click():
    set_count(get_count() + 1)
    render()
```

然後我們來模擬點擊兩次：

```python
handle_click()
handle_click()
```

輸出結果：

```python
Clicked! Count = 1
Clicked! Count = 2
```

恩？這不就是我們剛才在 React 做的事情嗎？

## OOP vs 函數式

你可能會問，那為什麼 React 不用 OOP？

用類別不是比較直覺嗎？像我們剛才用 Python 寫的 `Counter` 類別，狀態一目了然！

我們回到剛才的範例，看看這兩種思維在「狀態管理」上的實際差異。

用 Python OOP 寫計數器，這種寫法就像你在寫桌面應用程式，整個元件包在一個類別裡：

- `self.count` 是狀態。
- `handle_click` 是事件處理。
- `render` 是更新畫面。

優點是結構清楚、行為封裝好。

但問題來了：當這個元件越來越複雜時，你可能要管理好幾個狀態（例如倒數計時、背景色、錯誤提示等），這些都塞進 self，很快地，你的程式就會變這樣：

```python
self.count
self.timer
self.error_message
self.loading
self.is_dark_mode
self.is_visible
self.is_editable
self.is_submitting
self.is_valid
self.is_disabled
```

然後你開始懷疑人生。

## 函數式思維

React 用 Hook，實際上是把狀態邏輯抽出來變成一個個小函數，讓你可以「拼」出元件功能。

還記得我們的 `use_state` 嗎？

```python
get_count, set_count = use_state(0)
```

你現在有兩個函數 `get_count()` 和 `set_count()`，可以在任何地方使用，不需要一個類別來保存這些資訊。

**這也是 Hook 的精髓：狀態就被綁在那裡，只要你能拿到 `get_count()` 或 `set_count()`，就能隨時取得和更新它。就像拿著一個掛鉤 (Hook)，隨時可以勾住那段狀態，把它拉出來用。**

不信？那我們一起來試試看加個功能：「每五次點擊要提醒一次。」

- **用 OOP 寫**

  ```python
  class Counter:

      def __init__(self):
          self.count = 0

      def handle_click(self):
          self.count += 1
          if self.count % 5 == 0:
              print("🎉 你點了 5 次！")
          self.render()

      def render(self):
          print(f"Clicked! Count = {self.count}")
  ```

- **用函數式寫**

  ```python
  get_count, set_count = use_state(0)

  def use_celebrate_every(n):
      def check():
          if get_count() % n == 0 and get_count() != 0:
              print(f"🎉 你點了 {n} 次！")
      return check

  celebrate = use_celebrate_every(5)

  def handle_click():
      set_count(get_count() + 1)
      celebrate()
      render()

  def render():
      print(f"Clicked! Count = {get_count()}")
  ```

  這邊我們把「每 n 次觸發提醒」這件事拆成一個獨立的 Hook：`use_celebrate_every(n)`，你可以拿來給任何需要「定時提醒」的元件用。

  這就是 函數式的力量：你把功能變成一塊塊積木，拆解、重組、共用都更容易。

---

但函數式看起來就是比較複雜啊！

沒關係，我們再來看一個例子。

事實上，如果只是要寫一個簡單的按鈕，OOP 比函數式直觀太多了。

但 React 不是為了讓你「寫一個按鈕」，而是讓你能夠「組裝一整個應用」。

所以我們要問的其實不是：「誰比較簡單？」

而是：「**當事情變複雜的時候，誰還撐得住？**」

## 換個場景來想

假設你一開始寫一個簡單的元件，長這樣：

```jsx
function LoginForm() {
  const [username, setUsername] = useState("");
  return (
    <input value={username} onChange={(e) => setUsername(e.target.value)} />
  );
}
```

沒問題，乾淨簡單。

但接下來，產品經理來了，他說：

1. 登入時要顯示 loading 動畫。
2. 錯誤訊息要 3 秒後自動消失。
3. 記住使用者的帳號，儲存在 localStorage。
4. 輸入完要自動 focus 密碼欄位。
5. 若使用者連續失敗三次，鎖帳號 30 秒。

這時你要處理：

- loading 狀態
- 錯誤訊息顯示與自動清除
- 副作用：localStorage 操作、focus 控制
- 時間邏輯：鎖定計時器

## 如果用 OOP 寫？

你可能寫成這樣：

```python
class LoginForm:

    def __init__(self):
        self.username = ""
        self.loading = False
        self.error_message = ""
        self.failed_attempts = 0
        self.lock_until = None
        # 然後再寫一堆方法：handle_input、handle_submit、handle_error...

```

狀態塞在一起、邏輯散在十幾個方法裡，一言不合就爆炸？

你想重用「錯誤訊息消失倒數三秒」這個行為？對不起，那是 `LoginForm` 私有的，別人不能重用。你只能選擇重寫同樣功能的程式碼，或是直接繼承那個類別。

## 那如果用函數式呢？

你可以這樣拆：

```jsx
const [username, setUsername] = useState("");
const [loading, setLoading] = useState(false);
const errorMessage = useTimedError(3000); // 錯誤訊息顯示三秒
useLocalStorage("last-username", username); // 自動記帳號
useAutoFocus(ref); // 聚焦欄位
useLoginRateLimit(); // 失敗次數限制
```

每一個需求，都可以被封裝成一個「Hook 模組」，不會彼此糾纏。

你就像在拼積木，而不是解繩結。

是不是輕鬆很多？

## 為什麼要放棄 OOP？

其實 React 一開始是有 `class` 元件的，你可能還記得這種寫法：

```jsx
class Counter extends React.Component {
  state = { count: 0 };
  handleClick = () => this.setState({ count: this.state.count + 1 });
  render() {
    return (
      <button onClick={this.handleClick}>Count = {this.state.count}</button>
    );
  }
}
```

但當你開始有「共用邏輯」的需求時（像是驗證表單、偵測裝置大小、API 請求等），這些行為無法拆成獨立模組重用，你只能用「高階元件」或「傳入函式元件」來硬拆，結果愈寫愈亂，搞得自己像是在寫 Java？

抱歉我沒有要冒犯 Java 的意思。

## 所以

你說函數式看起來複雜？對，一開始就是。

但一個按鈕的複雜度和一個完整應用的複雜度，是兩回事。

**Hook 的設計，不是為了讓「入門更簡單」，而是讓「規模變大時不崩潰」。**

等你寫到第 8 個元件都要做「自動儲存」、「錯誤處理」、「視窗偵測」的時候，你就會慶幸自己是用 Hook，而不是一個全塞在物件裡的詭異生物。
