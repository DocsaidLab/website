---
slug: latex-usage
title: LaTeX 構文クイックリファレンス表
authors: Z. Yuan
image: /ja/img/2024/0604.webp
tags: [LaTeX, Math]
description: LaTeX 構文のクイックリファレンス
---

LaTeX を使用するたびに一部の構文を忘れることが多いため、このクイックリファレンス表に頻繁に使用する LaTeX 構文をまとめました。いつでも参照できるように便利な形で整理しています。

<!-- truncate -->

| **カテゴリ**         | **説明**                       |                    **構文**                    |                  **表示例**                   |
| -------------------- | ------------------------------ | :--------------------------------------------: | :-------------------------------------------: |
| **テキストスタイル** | 太字                           |                 `\textbf{AB}`                  |                 $\textbf{AB}$                 |
|                      | イタリック                     |                 `\textit{AB}`                  |                 $\textit{AB}$                 |
|                      | 下線                           |                `\underline{AB}`                |               $\underline{AB}$                |
|                      | 上線                           |                `\overline{AB}`                 |                $\overline{AB}$                |
|                      | タイプライター体               |                 `\texttt{AB}`                  |                 $\texttt{AB}$                 |
| **数学構造**         | 分数                           |                 `\frac{A}{B}`                  |                 $\frac{A}{B}$                 |
|                      | 分数（全行表示）               |                 `\dfrac{a}{b}`                 |                $\dfrac{a}{b}$                 |
|                      | 分数（行内表示）               |                 `\tfrac{a}{b}`                 |                $\tfrac{a}{b}$                 |
|                      | 分数（旧式）                   |                 `{A \over B}`                  |                 $\frac{A}{B}$                 |
|                      | 組合せ数                       |                 `\binom{n}{k}`                 |                $\binom{n}{k}$                 |
|                      | 組合せ数（旧式）               |                `{n \choose k}`                 |                $\binom{n}{k}$                 |
|                      | ルート                         |                   `\sqrt{x}`                   |                  $\sqrt{x}$                   |
|                      | n 次ルート                     |                 `\sqrt[n]{x}`                  |                 $\sqrt[n]{x}$                 |
|                      | 累乗                           |                     `a^b`                      |                     $a^b$                     |
|                      | 下付き文字                     |                     `a_b`                      |                     $a_b$                     |
|                      | 積分                           |          `\int_a^b x^2 \mathrm{d} x`           |          $\int_a^b x^2 \mathrm{d} x$          |
|                      | 総和                           |        `\sum_{n=1}^\infty \frac{1}{n}`         |        $\sum_{n=1}^\infty \frac{1}{n}$        |
|                      | 極限                           |       `\lim_{x \to \infty} \frac{1}{x}`        |       $\lim_{x \to \infty} \frac{1}{x}$       |
|                      | 積                             |               `\prod_{i=1}^n i`                |               $\prod_{i=1}^n i$               |
| **数学記号**         | サイン                         |                 `\sin{\theta}`                 |                $\sin{\theta}$                 |
|                      | コサイン                       |                 `\cos{\theta}`                 |                $\cos{\theta}$                 |
|                      | 正負符号                       |                     `\pm`                      |                     $\pm$                     |
|                      | 乗算記号                       |                    `\times`                    |                   $\times$                    |
|                      | 除算記号                       |                     `\div`                     |                    $\div$                     |
|                      | 包含関係                       |                  `\subseteq`                   |                  $\subseteq$                  |
|                      | 超集合関係                     |                  `\supseteq`                   |                  $\supseteq$                  |
|                      | 含意                           |                   `\implies`                   |                  $\implies$                   |
|                      | 被含意                         |                  `\impliedby`                  |                 $\impliedby$                  |
|                      | 同値                           |                     `\iff`                     |                    $\iff$                     |
|                      | 積集合                         |                     `\cap`                     |                    $\cap$                     |
|                      | 和集合                         |                     `\cup`                     |                    $\cup$                     |
|                      | 論理積                         |                    `\land`                     |                    $\land$                    |
|                      | 論理和                         |                     `\lor`                     |                    $\lor$                     |
|                      | 否定                           |                     `\neg`                     |                    $\neg$                     |
|                      | 不等号                         |                     `\neq`                     |                    $\neq$                     |
|                      | 近似                           |                   `\approx`                    |                   $\approx$                   |
| **ギリシャ文字**     | 小文字 α                       |                    `\alpha`                    |                   $\alpha$                    |
|                      | 大文字 Α                       |                    `\Alpha`                    |                   $\Alpha$                    |
|                      | 小文字 β                       |                    `\beta`                     |                    $\beta$                    |
|                      | 大文字 Β                       |                    `\Beta`                     |                    $\Beta$                    |
|                      | 小文字 γ                       |                    `\gamma`                    |                   $\gamma$                    |
|                      | 大文字 Γ                       |                    `\Gamma`                    |                   $\Gamma$                    |
|                      | 小文字 δ                       |                    `\delta`                    |                   $\delta$                    |
|                      | 大文字 Δ                       |                    `\Delta`                    |                   $\Delta$                    |
|                      | 小文字 ε                       |                   `\epsilon`                   |                  $\epsilon$                   |
|                      | 大文字 Ε                       |                   `\Epsilon`                   |                  $\Epsilon$                   |
|                      | バリアント小文字 φ             |                   `\varphi`                    |                   $\varphi$                   |
|                      | 大文字 Φ                       |                     `\Phi`                     |                    $\Phi$                     |
|                      | 小文字 χ                       |                     `\chi`                     |                    $\chi$                     |
|                      | 大文字 Χ                       |                     `\Chi`                     |                    $\Chi$                     |
|                      | 小文字 μ                       |                     `\mu`                      |                     $\mu$                     |
|                      | 大文字 Μ                       |                     `\Mu`                      |                     $\Mu$                     |
|                      | 小文字 ω                       |                    `\omega`                    |                   $\omega$                    |
|                      | 大文字 Ω                       |                    `\Omega`                    |                   $\Omega$                    |
| **行列とベクトル**   | 行列（括弧あり）               | `\begin{pmatrix} a & b \\ c & d \end{pmatrix}` | $\begin{pmatrix} a & b\\ c & d \end{pmatrix}$ |
|                      | 行列（括弧なし）               |  `\begin{matrix} x & y \\ z & w \end{matrix}`  | $\begin{matrix} x & y \\ z & w \end{matrix}$  |
|                      | ベクトル                       |                   `\vec{v}`                    |                   $\vec{v}$                   |
|                      | 単位行列                       |                  `\mathbf{I}`                  |                 $\mathbf{I}$                  |
|                      | 零行列                         |                  `\mathbf{0}`                  |                 $\mathbf{0}$                  |
| **その他**           | 角度                           |                    `\angle`                    |                   $\angle$                    |
|                      | 三角形                         |                  `\triangle`                   |                  $\triangle$                  |
|                      | 正方形                         |                   `\square`                    |                   $\square$                   |
|                      | 空白スペース                   |                    `\quad`                     |                    $\quad$                    |
|                      | 比例関係                       |                   `\propto`                    |                   $\propto$                   |
|                      | なぜなら                       |                   `\because`                   |                  $\because$                   |
|                      | したがって                     |                  `\therefore`                  |                 $\therefore$                  |
|                      | 整数集合                       |                  `\mathbb{Z}`                  |                 $\mathbb{Z}$                  |
|                      | 確率集合                       |                  `\mathbb{P}`                  |                 $\mathbb{P}$                  |
|                      | 実数集合                       |                  `\mathbb{R}`                  |                 $\mathbb{R}$                  |
|                      | 複素数集合                     |                  `\mathbb{C}`                  |                 $\mathbb{C}$                  |
|                      | 虚数空間                       |                     `\Im`                      |                     $\Im$                     |
|                      | 実数空間                       |                     `\Re`                      |                     $\Re$                     |
|                      | 空集合                         |                  `\emptyset`                   |                  $\emptyset$                  |
|                      | 空集合（美しい）               |                 `\varnothing`                  |                 $\varnothing$                 |
|                      | 属する                         |                     `\in`                      |                     $\in$                     |
|                      | 属さない                       |                   `\not\in`                    |                   $\not\in`                   |
|                      | 左回り円矢印                   |               `\circlearrowleft`               |              $\circlearrowleft$               |
|                      | 右回り円矢印                   |              `\circlearrowright`               |              $\circlearrowright$              |
|                      | プランク定数                   |                    `\hbar`                     |                    $\hbar$                    |
|                      | 自然対数                       |                     `\ln`                      |                     $\ln$                     |
|                      | 定数 $\pi$                     |                     `\pi`                      |                     $\pi$                     |
| **フォーマット**     | 関数の色                       |         `f(x) = a{\color{red}{x}} + b`         |        $f(x) = a{\color{red}{x}} + b$         |
|                      | カラーボックス                 |        `\colorbox{#eeeeee}{Color Box}`         |        $\colorbox{#eeeeee}{Color Box}$        |
|                      | カリグラフィー体               |            `{\cal ABCDE12345abced}`            |           ${\cal ABCDE12345abced}$            |
|                      | テキストボックス               |              `\fbox{boxed text}`               |              $\fbox{boxed text}$              |
|                      | 表示スタイルのテキストボックス |             `\boxed{boxed\ text}`              |             $\boxed{boxed\ text}$             |
|                      | ゴシック体                     |           `{\frak ABCDE12345abcde}`            |           ${\frak ABCDE12345abcde}$           |
|                      | イタリック                     |        `{\it abefg12345}\ abcdefg12345`        |       ${\it abefg12345}\ abcdefg12345$        |
|                      | 最小化                         |               `\min\limits_{n}`                |               $\min\limits_{n}$               |
|                      | 太字ギリシャ文字               |             `\boldsymbol{\alpha}`              |             $\boldsymbol{\alpha}$             |
