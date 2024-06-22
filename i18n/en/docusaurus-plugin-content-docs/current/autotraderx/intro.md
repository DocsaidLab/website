---
sidebar_position: 1
---

# Introduction

:::warning
As this project involves financial market trading, please read and comply with the [**disclaimer**](./disclaimer.md) carefully.
:::

This project primarily integrates with brokerage APIs for automated trading.

- [**AutoTraderX GitHub**](https://github.com/DocsaidLab/AutoTraderX)

![title](./img/title.webp)

## Overview

Recently, during a discussion with friends, we expressed interest in automating trading using brokerage APIs.

Coincidentally, we also wanted to apply deep learning models to practical financial trading, thus initiating this project.

We then consulted ChatGPT for a project name suggestion, and it proposed `AutoTraderX`. The rationale behind it was:

- 'Auto' signifies automation, 'Trader' denotes trading, and 'X' adds a touch of mystery. (Really?)

Who asked for the unnecessary mysterious vibes?

Nevertheless, appreciating its sincerity, we reluctantly accepted. (~ Actually, we quite liked it... ~)

## Automated Trading Platforms

Let's first review common automated trading platforms available:

- [**Quantitative Trading Platform Analysis, Quickly Find Your Suitable Program Trading Software and Platform! [TradingView, MultiCharts, Python, MT4]**](https://quantpass.org/software-comparison/)
- [**Recommended Program Trading Software | 4 Common Program Trading Tools: Multicharts, XQ, Python, and Excel VBA**](https://www.myrichfut.com/%E7%A8%8B%E5%BC%8F%E4%BA%A4%E6%98%93%E8%BB%9F%E9%AB%94%E6%8E%A8%E8%96%A6)
- [**Comparison of Four Major Quantitative Trading Platforms, Program Trading Strategy Sharing | Stock Stock Academy**](https://school.gugu.fund/blog/gugu_knowledge/7807989341)

After reviewing these summary articles, we can roughly summarize them in a table:

| Platform                             | Advantages                                                                    | Disadvantages                                                        |
| ------------------------------------ | ----------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **MultiCharts**                      | Simple syntax, intuitive and easy backtesting                                 | Requires integration on your own, advanced features are paid         |
| **Excel VBA**                        | Integration with other Excel functions and data sources                       | Lacks specialized financial market data and trading-related features |
| **Python**                           | High flexibility, extensive functionality, if you can code it, it can be done | You usually can't code it                                            |
| **XQ Quantitative Trading Platform** | Plenty of program samples like stock selection platforms                      | Low flexibility, advanced features are paid                          |

From this perspective, the only readily usable tool seems to be MultiCharts, but it's quite costly (starting at approximately $1000 NTD/month for the basic version).

Well, we're just rookies in program trading, just trying it out, so let's build it ourselves!

After all, engineers aren't afraid of being rookies, they're afraid of not trying.

## Project Goals

We have obtained a Python API from a brokerage, and now the story begins!

### Phase One Goals

Integrate with [**MasterLink's Python API**](https://mlapi.masterlink.com.tw/web_api/service/home) and automate trading using this API.

![masterlink](./img/masterlink.jpg)

### Phase Two Goals

After completing Phase One, we will consider integrating APIs from other brokerages and explore how to connect with them.
