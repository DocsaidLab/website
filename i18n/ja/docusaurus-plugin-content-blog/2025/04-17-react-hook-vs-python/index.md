---
slug: react-hook-vs-python
title: Reactは結局、何をHookしているのか？
authors: Z. Yuan
image: /ja/img/2025/0417.jpg
tags: [react, hook]
description: React Hookの背後にある概念を解説する。
---

数年前、React を学び始めたとき、React はまるで平行世界に存在しているかのようで、理解するのが難しかったです。

この Hook を見てください、これはあまりにも反人類的じゃないですか？

<!-- truncate -->

## Python から始めよう

あなたも私と同じように、Python の方が得意だと思うので、まずは Python から始めましょう！

まず最初に、最もシンプルなカウンターを見てみましょう。React では、次のように書きます：

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

もしこれを Python で書いたら、どう書きますか？

簡単だと思いますか？もちろん OOP で書きますよね！

## 伝統的な OOP の書き方

私はこんな風に書きます。この問題自体は難しくありません：

```python
class Counter:

    def __init__(self):
        self.count = 0

    def handle_click(self):
        self.count += 1
        self.render()

    def render(self):
        print(f"Clicked! Count = {self.count}")


# 使用例
counter = Counter()
counter.handle_click()  # 一回クリック
counter.handle_click()  # もう一回クリック
```

ですが、この書き方は上記の React とはなかなか比較しにくいので、別の書き方を試してみましょう：

## 関数型 + 状態のバインディング

クラスを使わず、直接関数を定義して、「状態」と「描画ロジック」を分け、純粋関数とクロージャを使って状態を管理します：

```python
def use_state(initial_value):

    state = {"value": initial_value}

    def get_value():
        return state["value"]

    def set_value(new_value):
        state["value"] = new_value

    return get_value, set_value
```

ここまで来ると、少し似てきた感じがしませんか？

次に、状態を定義します：

```python
get_count, set_count = use_state(0)
```

これで純粋な Python です。もう React を書いているわけではありません。

`get_count` と `set_count` という 2 つの関数を取得し、「クリックイベント」関数を書きましょう：

```python
def render():
    print(f"Clicked! Count = {get_count()}")

def handle_click():
    set_count(get_count() + 1)
    render()
```

そして、2 回クリックをシミュレーションします：

```python
handle_click()
handle_click()
```

出力結果：

```python
Clicked! Count = 1
Clicked! Count = 2
```

おや？これって、先程 React でやったことと同じじゃないですか？

## OOP vs 関数型

「なぜ React は OOP を使わないのか？」と疑問に思うかもしれません。

クラスを使う方が直感的ではないか？先ほど Python で書いた`Counter`クラスのように、状態は一目でわかりますよね！

もう一度、先ほどの例を振り返り、この 2 つの考え方が「状態管理」においてどのように異なるかを見てみましょう。

Python の OOP でカウンターを作る場合、この書き方はまるでデスクトップアプリケーションを書くようなものです。コンポーネント全体が 1 つのクラスの中に包まれています：

- `self.count` が状態。
- `handle_click` がイベント処理。
- `render` が画面更新。

利点は構造が明確で、動作がカプセル化されていることです。

しかし問題は、コンポーネントが複雑になるにつれて、複数の状態（例えばカウントダウン、背景色、エラーメッセージなど）を管理しなければならなくなります。これらをすべて`self`に詰め込んでいくと、すぐに次のような状態になってしまいます：

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

そして、あなたは人生に疑問を抱き始めるでしょう。

## 関数型思考

React は Hook を使用することで、状態のロジックを個別の小さな関数に分解し、コンポーネント機能を「組み立てる」ことができるようにしています。

私たちの`use_state`を覚えていますか？

```python
get_count, set_count = use_state(0)
```

これで、`get_count()`と`set_count()`の 2 つの関数が得られます。これらはクラスで情報を保持する必要なく、どこでも使用できます。

**これが Hook の本質です：状態はそこに固定されており、`get_count()`や`set_count()`さえあれば、いつでも取得して更新できるのです。まるでフック（Hook）を使ってその状態を引っ掛け、必要な時にそれを引き出すようなものです。**

信じられませんか？それなら、機能を追加してみましょう。「5 回クリックするごとにお知らせする」機能です。

- **OOP で書く場合**

  ```python
  class Counter:

      def __init__(self):
          self.count = 0

      def handle_click(self):
          self.count += 1
          if self.count % 5 == 0:
              print("🎉 5回クリックしました！")
          self.render()

      def render(self):
          print(f"Clicked! Count = {self.count}")
  ```

- **関数型で書く場合**

  ```python
  get_count, set_count = use_state(0)

  def use_celebrate_every(n):
      def check():
          if get_count() % n == 0 and get_count() != 0:
              print(f"🎉 {n}回クリックしました！")
      return check

  celebrate = use_celebrate_every(5)

  def handle_click():
      set_count(get_count() + 1)
      celebrate()
      render()

  def render():
      print(f"Clicked! Count = {get_count()}")
  ```

  ここでは、「n 回ごとにお知らせする」という機能を`use_celebrate_every(n)`という独立した Hook として分けました。この Hook は、定期的にお知らせが必要な他のコンポーネントにも使用できます。

  これが関数型の力です：機能をブロックとして分解し、再利用、再構成、共有がしやすくなるのです。

---

しかし、関数型は見た目が複雑すぎると感じるかもしれません！

心配しないで、次にもう一つの例を見てみましょう。

実際、もし単純なボタンを作るだけなら、OOP の方が関数型よりもずっと直感的です。

しかし、React は「ボタンを作るためにあるわけではありません」。React の目的は、「アプリケーション全体を組み立てる」ためにあるのです。

だから私たちが問うべきことは、実は「どちらが簡単か？」ではなく、「**物事が複雑になったとき、どちらが耐えられるか？**」なのです。

## 場面を変えて考えよう

仮に、最初にシンプルなコンポーネントを書いたとします。こんな感じで：

```jsx
function LoginForm() {
  const [username, setUsername] = useState("");
  return (
    <input value={username} onChange={(e) => setUsername(e.target.value)} />
  );
}
```

問題なし、シンプルで綺麗です。

しかし、次にプロダクトマネージャーが来て、こう言います：

1. ログイン時にローディングアニメーションを表示する。
2. エラーメッセージは 3 秒後に自動で消える。
3. ユーザーのアカウントを記憶し、localStorage に保存する。
4. 入力後、パスワードフィールドに自動でフォーカスを当てる。
5. ユーザーが連続して 3 回失敗した場合、アカウントを 30 秒間ロックする。

これで、次のことを処理しなければなりません：

- ローディング状態
- エラーメッセージの表示と自動消去
- 副作用：localStorage 操作、フォーカス制御
- 時間ロジック：ロックタイマー

## OOP で書いた場合は？

こんな風に書くかもしれません：

```python
class LoginForm:

    def __init__(self):
        self.username = ""
        self.loading = False
        self.error_message = ""
        self.failed_attempts = 0
        self.lock_until = None
        # そして一堆のメソッドを書いていく：handle_input、handle_submit、handle_error...
```

状態が一緒に詰め込まれ、ロジックが 10 数個のメソッドに散らばっていく。気がつけば爆発しそうです。

「エラーメッセージが 3 秒後に消える」機能を再利用したい？ごめんなさい、それは`LoginForm`のプライベートな部分なので他の場所では使えません。同じ機能を再実装するか、そのクラスを継承するしかありません。

## では関数型の場合は？

こんな風に分けることができます：

```jsx
const [username, setUsername] = useState("");
const [loading, setLoading] = useState(false);
const errorMessage = useTimedError(3000); // エラーメッセージを3秒表示
useLocalStorage("last-username", username); // アカウントを自動で保存
useAutoFocus(ref); // 自動でフィールドにフォーカス
useLoginRateLimit(); // 失敗回数制限
```

それぞれの要求は、独立した「Hook モジュール」としてカプセル化され、お互いに干渉しません。

まるで積み木を組み立てるようで、結び目を解くような作業ではありません。

だいぶ楽になった感じがしませんか？

## なぜ OOP を放棄するのか？

実は、React には最初`class`コンポーネントがありました。この書き方を覚えているかもしれません：

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

しかし、「共通のロジック」が必要になった時（例えばフォームの検証、デバイスサイズの検出、API リクエストなど）、これらの処理は独立したモジュールとして再利用することができません。仕方なく「高階コンポーネント」や「関数コンポーネントへの渡し方」を使って無理やり分けることになります。結果、コードがますます複雑になり、まるで Java を書いているかのように感じるかもしれません。

※Java をディスっているわけではありません。

## というわけで

関数型は複雑に見える？はい、最初はそうです。

しかし、1 つのボタンの複雑さと、完全なアプリケーションの複雑さは、まったく異なる問題です。

**Hook の設計は「初心者向け」ではなく、「規模が大きくなっても崩れないようにする」ためのものです。**

あなたが 8 個目のコンポーネントに「自動保存」「エラーハンドリング」「ウィンドウ検出」を実装しなければならない時が来たとき、Hook を使っていたことを幸運に感じるでしょう。あの奇妙なクラスの中にすべてを詰め込むのではなく。
