---
slug: colorful-cli-with-ansi-escape-codes
title: 終端機不該只有黑與白
authors: Z. Yuan
image: /img/2025/0412.jpg
tags: [cli, python, ansi]
description: 帶你認識 ANSI Escape Code，讓 CLI 輸出也能走色彩繽紛系。
---

在寫 CLI 工具或 debug 輸出時，你是否曾懷疑人生：

> **「我剛剛 print 的這坨資訊，為什麼看起來像一碗沒有配料的陽春麵？」**

抱歉，我沒有要冒犯陽春麵的意思，我只是想說：

如果你也想讓終端機變得既有辨識性、又能一眼看到重點，那你需要認識一個低調卻強大的主角：

- **ANSI Escape Code**。

<!-- truncate -->

## 那是什麼？

ANSI，全文叫做 **American National Standards Institute**，這個組織負責制定各種標準，包括電腦的字元編碼。

ANSI Escape Code 是一組看起來很像亂碼的字元序列，這些序列長這樣：`\033[` 或 `\x1b[` 開頭，後面接上各種控制碼。

舉個例子：

```python
print("\033[1;31mHello World\033[0m")
```

上面這段程式會輸出一行紅色粗體字「Hello World」，最後的 `\033[0m` 是 reset，防止你後面的輸出也被染紅。

## 常見格式控制碼

你可以用這些數值讓文字「穿」上各種樣式：

<div style={{
  whiteSpace: 'nowrap',
  overflowX: 'auto',
  fontSize: '1rem',
  lineHeight: '0.8',
  justifyContent: 'center',
  display: 'flex',
}}>

| 數值 | 效果              |
| :--: | ----------------- |
|  0   | Reset（素顏）     |
|  1   | Bold（粗體）      |
|  2   | Dim（低調）       |
|  3   | Italic（斜一點）  |
|  4   | Underline（底線） |
|  7   | Invert（反轉色）  |

</div>

這些可以與顏色搭配使用，打造屬於你 CLI 的「個性穿搭」。

:::info
**ANSI 控制碼冷知識：5 跟 6 去哪了？**

其實它們一直都在，只是大部分人看不到。

- **5 = 慢閃（slow blink）**
- **6 = 快閃（rapid blink）**

不過……

**幾乎沒終端機支援它們**，尤其現代系統為了避免眼睛爆炸，幾乎都把 blink 效果關閉了。

你可以試試看：

```python
print("\033[5;31m這是一段慢閃的紅字\033[0m")
```

如果它真的閃了，恭喜你，你的終端還活在上古年代 😅
:::

## 顏色控制碼

ANSI Escape Code 中的文字顏色代碼通常為 30–37（標準色）與 90–97（亮色系）。

以下為對應表：

<div style={{
  whiteSpace: 'nowrap',
  overflowX: 'auto',
  fontSize: '1rem',
  lineHeight: '0.8',
  justifyContent: 'center',
  display: 'flex',
}}>

| 數值  | 顏色     |
| :---: | -------- |
|  30   | 黑色     |
|  31   | 紅色     |
|  32   | 綠色     |
|  33   | 黃色     |
|  34   | 藍色     |
|  35   | 品紅     |
|  36   | 青色     |
|  37   | 白色     |
| 90–97 | 亮色系列 |

</div>

而背景色控制碼則對應為 40–47（標準色）與 100–107（亮色系），也可同樣類推如下：

<div style={{
  whiteSpace: 'nowrap',
  overflowX: 'auto',
  fontSize: '1rem',
  lineHeight: '0.8',
  justifyContent: 'center',
  display: 'flex',
}}>

|  數值   | 背景顏色           |
| :-----: | ------------------ |
|   40    | 黑色背景 (Black)   |
|   41    | 紅色背景 (Red)     |
|   42    | 綠色背景 (Green)   |
|   43    | 黃色背景 (Yellow)  |
|   44    | 藍色背景 (Blue)    |
|   45    | 品紅背景 (Magenta) |
|   46    | 青色背景 (Cyan)    |
|   47    | 白色背景 (White)   |
| 100–107 | 對應亮色背景版     |

</div>

---

**你可能會發現，`38` 到 `89` 之間幾乎沒人提 ── 那它們去哪了？**

- **38 與 48：自定義色彩的進階用法**

  - `38` = 開啟「自定義前景色」模式
  - `48` = 開啟「自定義背景色」模式

    這時候你需要再加參數指定顏色，例如 256 色（8-bit 模式）：

    ```python
    print("\033[38;5;198m一段粉紅色文字\033[0m")
    ```

    - `38;5;198` 代表使用 256 色調色盤中的第 198 號色。

    或是使用 True Color（24-bit 模式）：

    ```python
    print("\033[38;2;255;105;180mHello in pink!\033[0m")
    ```

    - `38;2;<r>;<g>;<b>` 讓你精準指定 RGB 值。

- **39 與 49：reset 顏色的好朋友**

  - `39`：還原預設前景色
  - `49`：還原預設背景色

:::tip
**那 50 ～ 89 呢？**

它們什麼都不是！

這段數值在 ANSI 規範裡**沒有明確定義**，屬於保留區或歷史空洞。少部分終端（例如某些 Konsole / xterm）有實驗性使用過，但都不是標準功能。

你硬塞這些數值下去，大多數終端不會理你。
:::

所以，我們整理一下，色碼地圖大概就是這樣：

<div style={{
  whiteSpace: 'nowrap',
  overflowX: 'auto',
  fontSize: '1rem',
  lineHeight: '0.8',
  justifyContent: 'center',
  display: 'flex',
}}>

|   區間    | 功能                       |
| :-------: | -------------------------- |
|   30–37   | 標準前景色                 |
|   90–97   | 亮色前景                   |
|  **38**   | 進階前景色（256 色 / RGB） |
|  **39**   | 還原前景色                 |
|   40–47   | 標準背景色                 |
|  100–107  | 亮色背景                   |
|  **48**   | 進階背景色（256 色 / RGB） |
|  **49**   | 還原背景色                 |
| **50–89** | ⚠️ 保留區，請勿輕易觸碰    |

</div>

寫一段程式來玩玩看吧：

```python
for i in range(256):
    print(f"\033[48;5;{i}m {i:3} \033[0m", end=" ")
    if i % 16 == 15:
        print()
```

執行後大概長得像這樣：

import Ansi256Grid from '@site/src/components/Ansi256Grid';

<Ansi256Grid />

## 組合語法範例

使用語法格式：

```
\033[<格式>;<文字顏色>m
```

範例：

```python
def colorstr(text, fmt=1, color=34):
    return f"\033[{fmt};{color}m{text}\033[0m"

print(colorstr("這是一段藍色粗體文字"))
```

更潮一點：

```python
print("\033[1;34;43m藍字配黃底，時尚不解釋\033[0m")
```

## 注意事項

使用時有幾個部分要注意，別讓你的彩色變災難：

- **一定要 reset**：沒加 `\033[0m`，你的輸出就會一路繼承下去，從紅色悲劇變成紅色長篇小說。
- **Jupyter 限制多**：能顯示顏色，但游標或畫面控制基本是夢。
- **Windows Terminal 已升級**：新版終於有良心，舊版就用 `colorama` 吧。
- **Log 檔案別亂塞**：除非你喜歡用正則表達式懲罰自己，不然建議儲存純文字版本。

---

## 延伸應用

你可以封裝自己的 color 工具模組，還能用 Enum 寫出更清爽的呼叫方式：

```python
from enum import Enum

class COLOR(Enum):
    RED = 31
    GREEN = 32
    BLUE = 34

class FORMAT(Enum):
    BOLD = 1
    UNDERLINE = 4

def colorstr(obj, color=COLOR.BLUE, fmt=FORMAT.BOLD):
    return f"\033[{fmt.value};{color.value}m{obj}\033[0m"
```

應用場景：

- **錯誤與警告提示**：用紅色說「這邊有問題」。
- **成功訊息**：綠色的 Checkmark 就是爽。
- **互動選單提示**：讓使用者知道該輸入什麼。

---

## 不想自己處理？

很多套件已幫你打包好了，我們可以直接拿來用。

參考以下幾個 Python 套件：

1. **`colorama`**

   解決跨平台顯示，Windows 小幫手，用法簡單：

   ```python
   from colorama import init, Fore, Style
   init()
   print(Fore.RED + "Hello in red!" + Style.RESET_ALL)
   ```

2. **`termcolor`**

   提供 `colored()`，不囉嗦：

   ```python
   from termcolor import colored
   print(colored("Warning", "red", attrs=["bold"]))
   ```

3. **`rich`**

   支援顏色、表格、進度條、Markdown，說它是 CLI 的 React 也不為過。

   ```python
   from rich import print
   print("[bold red]This is a bold red text[/bold red]")
   ```

## 小結

掌握 ANSI Escape Code，就像給你的終端加上調色盤。

從 Debug 到 CLI 工具開發，你可以用最少的工具，做出最顯眼的效果。

更重要的是，讓使用者一眼看出「紅的是錯」、「綠的是過」、「黃的是等等再看」，不再迷失在一片黑白訊息中。

讓終端機的世界繽紛起來，從 `\033[` 開始。
