---
slug: latex-usage
title: LaTeX Syntax Quick Reference
authors: Z. Yuan
image: /en/img/2024/0604.webp
tags: [LaTeX, Math]
description: Quick reference guide for LaTeX syntax
---

Every time you need it, it's always a hassle to find, so we decided to write a LaTeX syntax quick reference table.

<!-- truncate -->

| **Category**           | **Description**          |                   **Syntax**                   |                  **Display**                  |
| ---------------------- | ------------------------ | :--------------------------------------------: | :-------------------------------------------: |
| **Text Styles**        | Bold                     |                 `\textbf{AB}`                  |                 $\textbf{AB}$                 |
|                        | Italic                   |                 `\textit{AB}`                  |                 $\textit{AB}$                 |
|                        | Underline                |                `\underline{AB}`                |               $\underline{AB}$                |
|                        | Overline                 |                `\overline{AB}`                 |                $\overline{AB}$                |
|                        | Typewriter Font          |                 `\texttt{AB}`                  |                 $\texttt{AB}$                 |
| **Math Structures**    | Fraction                 |                 `\frac{A}{B}`                  |                 $\frac{A}{B}$                 |
|                        | Fraction (Display)       |                 `\dfrac{a}{b}`                 |                $\dfrac{a}{b}$                 |
|                        | Fraction (Inline)        |                 `\tfrac{a}{b}`                 |                $\tfrac{a}{b}$                 |
|                        | Old-Style Fraction       |                 `{A \over B}`                  |                 $\frac{A}{B}$                 |
|                        | Binomial Coeff.          |                 `\binom{n}{k}`                 |                $\binom{n}{k}$                 |
|                        | Old-Style Binomial       |                `{n \choose k}`                 |                $\binom{n}{k}$                 |
|                        | Square Root              |                   `\sqrt{x}`                   |                  $\sqrt{x}$                   |
|                        | n-th Root                |                 `\sqrt[n]{x}`                  |                 $\sqrt[n]{x}$                 |
|                        | Exponent                 |                     `a^b`                      |                     $a^b$                     |
|                        | Subscript                |                     `a_b`                      |                     $a_b$                     |
|                        | Integral                 |          `\int_a^b x^2 \mathrm{d} x`           |          $\int_a^b x^2 \mathrm{d} x$          |
|                        | Summation                |        `\sum_{n=1}^\infty \frac{1}{n}`         |        $\sum_{n=1}^\infty \frac{1}{n}$        |
|                        | Limit                    |       `\lim_{x \to \infty} \frac{1}{x}`        |       $\lim_{x \to \infty} \frac{1}{x}$       |
|                        | Product                  |               `\prod_{i=1}^n i`                |               $\prod_{i=1}^n i$               |
| **Math Symbols**       | Sine                     |                 `\sin{\theta}`                 |                $\sin{\theta}$                 |
|                        | Cosine                   |                 `\cos{\theta}`                 |                $\cos{\theta}$                 |
|                        | Plus-Minus               |                     `\pm`                      |                     $\pm$                     |
|                        | Multiplication           |                    `\times`                    |                   $\times$                    |
|                        | Division                 |                     `\div`                     |                    $\div$                     |
|                        | Subset Equal             |                  `\subseteq`                   |                  $\subseteq$                  |
|                        | Superset Equal           |                  `\supseteq`                   |                  $\supseteq`                  |
|                        | Implies                  |                   `\implies`                   |                  $\implies$                   |
|                        | Implied By               |                  `\impliedby`                  |                 $\impliedby$                  |
|                        | If and Only If           |                     `\iff`                     |                    $\iff$                     |
|                        | Intersection             |                     `\cap`                     |                    $\cap$                     |
|                        | Union                    |                     `\cup`                     |                    $\cup$                     |
|                        | Logical And              |                    `\land`                     |                    $\land$                    |
|                        | Logical Or               |                     `\lor`                     |                    $\lor$                     |
|                        | Logical Not              |                     `\neg`                     |                    $\neg$                     |
|                        | Not Equal To             |                     `\neq`                     |                    $\neq$                     |
|                        | Approximately Equal      |                   `\approx`                    |                   $\approx$                   |
| **Greek Letters**      | Lowercase α              |                    `\alpha`                    |                   $\alpha$                    |
|                        | Uppercase Α              |                    `\Alpha`                    |                   $\Alpha$                    |
|                        | Lowercase β              |                    `\beta`                     |                    $\beta$                    |
|                        | Uppercase Β              |                    `\Beta`                     |                    $\Beta$                    |
|                        | Lowercase γ              |                    `\gamma`                    |                   $\gamma$                    |
|                        | Uppercase Γ              |                    `\Gamma`                    |                   $\Gamma$                    |
|                        | Lowercase δ              |                    `\delta`                    |                   $\delta$                    |
|                        | Uppercase Δ              |                    `\Delta`                    |                   $\Delta$                    |
|                        | Lowercase ε              |                   `\epsilon`                   |                  $\epsilon$                   |
|                        | Uppercase Ε              |                   `\Epsilon`                   |                  $\Epsilon$                   |
|                        | Variant Lowercase φ      |                   `\varphi`                    |                   $\varphi$                   |
|                        | Uppercase Φ              |                     `\Phi`                     |                    $\Phi$                     |
|                        | Lowercase χ              |                     `\chi`                     |                    $\chi$                     |
|                        | Uppercase Χ              |                     `\Chi`                     |                    $\Chi$                     |
|                        | Lowercase μ              |                     `\mu`                      |                     $\mu$                     |
|                        | Uppercase Μ              |                     `\Mu`                      |                     $\Mu$                     |
|                        | Lowercase ω              |                    `\omega`                    |                   $\omega$                    |
|                        | Uppercase Ω              |                    `\Omega`                    |                   $\Omega$                    |
| **Matrices & Vectors** | Matrix (with brackets)   | `\begin{pmatrix} a & b \\ c & d \end{pmatrix}` | $\begin{pmatrix} a & b\\ c & d \end{pmatrix}$ |
|                        | Matrix (no brackets)     |  `\begin{matrix} x & y \\ z & w \end{matrix}`  | $\begin{matrix} x & y \\ z & w \end{matrix}$  |
|                        | Vector                   |                   `\vec{v}`                    |                   $\vec{v}$                   |
|                        | Identity Matrix          |                  `\mathbf{I}`                  |                 $\mathbf{I}$                  |
|                        | Zero Matrix              |                  `\mathbf{0}`                  |                 $\mathbf{0}$                  |
| **Miscellaneous**      | Angle                    |                    `\angle`                    |                   $\angle$                    |
|                        | Triangle                 |                  `\triangle`                   |                  $\triangle$                  |
|                        | Square                   |                   `\square`                    |                   $\square$                   |
|                        | Space                    |                    `\quad`                     |                    $\quad$                    |
|                        | Proportional To          |                   `\propto`                    |                   $\propto$                   |
|                        | Because                  |                   `\because`                   |                  $\because$                   |
|                        | Therefore                |                  `\therefore`                  |                 $\therefore$                  |
|                        | Integer Set              |                  `\mathbb{Z}`                  |                 $\mathbb{Z}$                  |
|                        | Probability Set          |                  `\mathbb{P}`                  |                 $\mathbb{P}$                  |
|                        | Real Number Set          |                  `\mathbb{R}`                  |                 $\mathbb{R}$                  |
|                        | Complex Number Set       |                  `\mathbb{C}`                  |                 $\mathbb{C}$                  |
|                        | Imaginary Part           |                     `\Im`                      |                     $\Im$                     |
|                        | Real Part                |                     `\Re`                      |                     $\Re`                     |
|                        | Empty Set                |                  `\emptyset`                   |                  $\emptyset$                  |
|                        | Fancy Empty Set          |                 `\varnothing`                  |                 $\varnothing$                 |
|                        | Element Of               |                     `\in`                      |                     $\in$                     |
|                        | Not Element Of           |                   `\not\in`                    |                   $\not\in$                   |
|                        | Counterclockwise Arrow   |               `\circlearrowleft`               |              $\circlearrowleft$               |
|                        | Clockwise Arrow          |              `\circlearrowright`               |              $\circlearrowright$              |
|                        | Planck's Constant        |                    `\hbar`                     |                    $\hbar$                    |
|                        | Natural Log              |                     `\ln`                      |                     $\ln$                     |
|                        | Constant $\pi$           |                     `\pi`                      |                     $\pi$                     |
| **Formatting**         | Function Color           |         `f(x) = a{\color{red}{x}} + b`         |        $f(x) = a{\color{red}{x}} + b$         |
|                        | Color Box                |        `\colorbox{#eeeeee}{Color Box}`         |        $\colorbox{#eeeeee}{Color Box}$        |
|                        | Calligraphic Font        |            `{\cal ABCDE12345abced}`            |           ${\cal ABCDE12345abced}$            |
|                        | Framed Text              |              `\fbox{boxed text}`               |              $\fbox{boxed text}$              |
|                        | Boxed Display Style Text |             `\boxed{boxed\ text}`              |             $\boxed{boxed\ text}$             |
|                        | Fraktur Font             |           `{\frak ABCDE12345abcde}`            |           ${\frak ABCDE12345abcde}$           |
|                        | Italic Font              |        `{\it abefg12345}\ abcdefg12345`        |       ${\it abefg12345}\ abcdefg12345$        |
|                        | Minimum                  |               `\min\limits_{n}`                |               $\min\limits_{n}$               |
|                        | Bold Greek Letter        |             `\boldsymbol{\alpha}`              |             $\boldsymbol{\alpha}$             |
