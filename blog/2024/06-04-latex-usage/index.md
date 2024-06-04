---
slug: latex-usage
title: LaTeX 語法快速查詢表
authors: Zephyr
image: /img/2024/0604.webp
tags: [LaTeX, Math]
description: LaTeX syntax quick reference
---

<figure>
![title](/img/2024/0604.webp)
<figcaption>封面圖片：由 GPT-4 閱讀本文之後自動生成</figcaption>
</figure>

---

每次要用的時候都是一陣好找，所以我們決定寫一份 LaTeX 語法快速查詢表格。

| **類別**       | **中文說明**       |                    **語法**                    |                 **顯示效果**                  |
| -------------- | ------------------ | :--------------------------------------------: | :-------------------------------------------: |
| **文字樣式**   | 加粗               |                 `\textbf{AB}`                  |                 $\textbf{AB}$                 |
|                | 斜體               |                 `\textit{AB}`                  |                 $\textit{AB}$                 |
|                | 底線               |                `\underline{AB}`                |               $\underline{AB}$                |
|                | 上劃線             |                `\overline{AB}`                 |                $\overline{AB}$                |
|                | 粗體向量符號       |                 `\mathbf{AB}`                  |                 $\mathbf{AB}$                 |
|                | 打字機字體         |                 `\texttt{AB}`                  |                 $\texttt{AB}$                 |
| **數學結構**   | 分數               |                 `\frac{A}{B}`                  |                 $\frac{A}{B}$                 |
|                | 分數（顯示整行）   |                 `\dfrac{a}{b}`                 |                $\dfrac{a}{b}$                 |
|                | 分數（顯示於行中） |                 `\tfrac{a}{b}`                 |                $\tfrac{a}{b}$                 |
|                | 分數（舊式）       |                 `{A \over B}`                  |                 $\frac{A}{B}$                 |
|                | 組合數             |                 `\binom{n}{k}`                 |                $\binom{n}{k}$                 |
|                | 組合數（舊式）     |                `{n \choose k}`                 |                $\binom{n}{k}$                 |
|                | 根號               |                   `\sqrt{x}`                   |                  $\sqrt{x}$                   |
|                | n 次根號           |                 `\sqrt[n]{x}`                  |                 $\sqrt[n]{x}$                 |
|                | 冪                 |                     `a^b`                      |                     $a^b$                     |
|                | 下標               |                     `a_b`                      |                     $a_b$                     |
|                | 積分               |          `\int_a^b x^2 \mathrm{d} x`           |          $\int_a^b x^2 \mathrm{d} x$          |
|                | 求和               |        `\sum_{n=1}^\infty \frac{1}{n}`         |        $\sum_{n=1}^\infty \frac{1}{n}$        |
|                | 極限               |       `\lim_{x \to \infty} \frac{1}{x}`        |       $\lim_{x \to \infty} \frac{1}{x}$       |
|                | 乘積               |               `\prod_{i=1}^n i`                |               $\prod_{i=1}^n i$               |
| **數學符號**   | 正弦               |                 `\sin{\theta}`                 |                $\sin{\theta}$                 |
|                | 餘弦               |                 `\cos{\theta}`                 |                $\cos{\theta}$                 |
|                | 正負號             |                     `\pm`                      |                     $\pm$                     |
|                | 乘號               |                    `\times`                    |                   $\times$                    |
|                | 除號               |                     `\div`                     |                    $\div$                     |
|                | 子集等於           |                  `\subseteq`                   |                  $\subseteq$                  |
|                | 超集等於           |                  `\supseteq`                   |                  $\supseteq$                  |
|                | 蘊涵               |                   `\implies`                   |                  $\implies$                   |
|                | 被蘊涵             |                  `\impliedby`                  |                 $\impliedby$                  |
|                | 當且僅當           |                     `\iff`                     |                    $\iff$                     |
|                | 交集               |                     `\cap`                     |                    $\cap$                     |
|                | 聯集               |                     `\cup`                     |                    $\cup$                     |
|                | 邏輯與             |                    `\land`                     |                    $\land$                    |
|                | 邏輯或             |                     `\lor`                     |                    $\lor$                     |
|                | 邏輯非             |                     `\neg`                     |                    $\neg$                     |
|                | 不等於             |                     `\neq`                     |                    $\neq$                     |
|                | 約等於             |                   `\approx`                    |                   $\approx$                   |
| **特殊符號**   | 希臘字母 α         |                    `\alpha`                    |                   $\alpha$                    |
|                | 希臘字母 β         |                    `\beta`                     |                    $\beta$                    |
|                | 希臘字母 γ         |                    `\gamma`                    |                   $\gamma$                    |
|                | 角                 |                    `\angle`                    |                   $\angle$                    |
|                | 三角形             |                  `\triangle`                   |                  $\triangle$                  |
|                | 正方形             |                   `\square`                    |                   $\square$                   |
|                | 實部               |                     `\Re`                      |                     $\Re$                     |
|                | 虛部               |                     `\Im`                      |                     $\Im$                     |
|                | 虛數單位 i         |                    `\imath`                    |                   $\imath$                    |
|                | 虛數單位 j         |                    `\jmath`                    |                   $\jmath$                    |
|                | 希臘字母 ε         |                   `\epsilon`                   |                  $\epsilon$                   |
|                | 希臘字母 φ         |                     `\phi`                     |                    $\phi$                     |
|                | 變形希臘字母 φ     |                   `\varphi`                    |                   $\varphi$                   |
| **矩陣和向量** | 矩陣（括號）       | `\begin{pmatrix} a & b \\ c & d \end{pmatrix}` | $\begin{pmatrix} a & b\\ c & d \end{pmatrix}$ |
|                | 矩陣（無括號）     |  `\begin{matrix} x & y \\ z & w \end{matrix}`  | $\begin{matrix} x & y \\ z & w \end{matrix}$  |
|                | 向量               |                   `\vec{v}`                    |                   $\vec{v}$                   |
|                | 單位矩陣           |                  `\mathbf{I}`                  |                 $\mathbf{I}$                  |
|                | 零矩陣             |                  `\mathbf{0}`                  |                 $\mathbf{0}$                  |
| **雜項**       | 角                 |                    `\angle`                    |                   $\angle$                    |
|                | 三角形             |                  `\triangle`                   |                  $\triangle$                  |
|                | 正方形             |                   `\square`                    |                   $\square$                   |
|                | 空格               |                    `\quad`                     |                    $\quad$                    |
|                | 正比               |                   `\propto`                    |                   $\propto$                   |
|                | 因為               |                   `\because`                   |                  $\because$                   |
|                | 所以               |                  `\therefore`                  |                 $\therefore$                  |
|                | 整數集             |                  `\mathbb{Z}`                  |                 $\mathbb{Z}$                  |
|                | 機率集             |                  `\mathbb{P}`                  |                 $\mathbb{P}$                  |
|                | 實數集             |                  `\mathbb{R}`                  |                 $\mathbb{R}$                  |
|                | 複數集             |                  `\mathbb{C}`                  |                 $\mathbb{C}$                  |
|                | 虛數空間           |                     `\Im`                      |                     $\Im$                     |
|                | 實數空間           |                     `\Re`                      |                     $\Re$                     |
|                | 空集合             |                  `\emptyset`                   |                  $\emptyset$                  |
|                | 空集合（好看）     |                 `\varnothing`                  |                 $\varnothing$                 |
|                | 屬於               |                     `\in`                      |                     $\in$                     |
|                | 不屬於             |                   `\not\in`                    |                   $\not\in$                   |
|                | 希臘字母 χ         |                     `\chi`                     |                    $\chi$                     |
|                | 逆時針圓箭頭       |               `\circlearrowleft`               |              $\circlearrowleft$               |
|                | 順時針圓箭頭       |              `\circlearrowright`               |              $\circlearrowright$              |
|                | 逆時針曲線箭頭     |               `\curvearrowleft`                |               $\curvearrowleft$               |
|                | 順時針曲線箭頭     |               `\curvearrowright`               |              $\curvearrowright$               |
|                | 普朗克常數         |                    `\hbar`                     |                    $\hbar$                    |
|                | 自然對數           |                     `\ln`                      |                     $\ln$                     |
|                | 常數 $\pi$         |                     `\pi`                      |                     $\pi$                     |
|                | 集合符號 $\cup$    |                     `\cup`                     |                    $\cup$                     |
|                | 集合符號 $\cap$    |                     `\cap`                     |                    $\cap$                     |
| **格式**       | 函數顏色           |         `f(x) = a{\color{red}{x}} + b`         |        $f(x) = a{\color{red}{x}} + b$         |
|                | 顏色框             |        `\colorbox{#eeeeee}{Color Box}`         |        $\colorbox{#eeeeee}{Color Box}$        |
|                | 書法字體           |            `{\cal ABCDE12345abced}`            |           ${\cal ABCDE12345abced}$            |
|                | 框內文本           |              `\fbox{boxed text}`               |              $\fbox{boxed text}$              |
|                | 顯示風格框內文本   |             `\boxed{boxed\ text}`              |             $\boxed{boxed\ text}$             |
|                | 黑體字             |           `{\frak ABCDE12345abcde}`            |           ${\frak ABCDE12345abcde}$           |
|                | 斜體               |        `{\it abefg12345}\ abcdefg12345`        |       ${\it abefg12345}\ abcdefg12345$        |
|                | 最小化             |               `\min\limits_{n}`                |               $\min\limits_{n}$               |
|                | 粗體希臘字母       |             `\boldsymbol{\alpha}`              |             $\boldsymbol{\alpha}$             |
