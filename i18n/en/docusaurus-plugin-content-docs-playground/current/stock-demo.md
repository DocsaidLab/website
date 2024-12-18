# StockAnalysis Demo

Everyone has some experience with stock investment, and stock analysis is an essential part of the investment decision-making process.

However, stock analysis is a vast and multidimensional task. Considering the limited time and resources, we have divided this task into several stages, gradually improving and constructing a comprehensive stock analysis tool.

:::info
**There are a few things to note when using this webpage:**

1. Please read the [**Disclaimer**](#-disclaimer) first to ensure you have a correct understanding of the usage of this webpage.
2. The FinMind API has usage limits. For details, refer to: [**Usage Limit Explanation**](#-usage-limits).
3. We do not intend to rebuild a candlestick chart analysis tool. The earlier stages are preparations for the final modeling analysis, and no additional indicators may be added.
4. Historical transaction data comes from the FinMind API, and the values may include after-market transactions, so the volume may differ from other websites.
5. Currently, only Taiwan stock market data is supported, and stock data from other markets will be addressed in future versions.
   :::

## Stage 1: Basic Analysis

In the first stage, we focus on the basic stock price and trading volume analysis. Through visualization, we help users grasp the fundamentals and technical trends of the stocks. The tasks involved are roughly as follows:

1.  **Connecting Data Sources**: Using FinMind as the data source for stock information, obtaining historical stock price data of Taiwan stocks through the API.
2.  **Chart Visualization**

    - Transform raw data into easy-to-understand charts, including line charts, bar charts, and candlestick charts.
    - Add interactive features, allowing users to adjust the time range and select individual stocks.

3.  **Technical Indicator Calculation**

    - Calculate and present common technical indicators:
      - **Bollinger Bands**: Shows the price fluctuation range and moving averages.
      - **MACD** (Moving Average Convergence Divergence): Identifies market momentum and trend reversal signals.
      - **RSI** (Relative Strength Index): Determines overbought or oversold conditions in the market.
      - **KDJ** Indicator: Reflects the momentum of stock price fluctuations and short-term buying and selling opportunities.

Through charts and indicators, users can quickly understand the historical price and volume performance of a stock and apply basic candlestick theory to make preliminary buy and sell judgments.

### Program Features

import StockAnalysisPage from '@site/src/components/StockAnalysis';

<StockAnalysisPage />

## Stage 2: News Analysis

By combining real-time news in the stock market with price trends, we extract semantic information to analyze the potential impact of news on short-term stock movements.

The planned tasks include:

1. **Integration of News Data Sources**: Connect to news platform APIs, such as Google News, Yahoo Finance News, FinMind, etc., to obtain real-time stock-related news and reports.
2. **Semantic Feature Extraction**: Integrate OpenAI models for news sentiment analysis, extracting features such as sentiment, themes, and importance from news reports.
3. **Cross-comparison of News and Stock Trends**
   - Align news reports with changes in stock prices and trading volumes over time, observing correlations between the two.
   - Use statistical data and visual charts to display the potential impact of news sentiment on short-term stock price fluctuations.

The analysis results will reveal the relationship between news sentiment and stock price trends, helping users gain insights into market reactions and identify potential market opportunities or risks.

### 🚧 Program Features 🚧

(Completion timeline pending)

## Stage 3: Modeling Analysis

Since deep learning techniques are often used to solve problems, there’s no reason not to apply them to stock analysis.

In this stage, we will introduce deep learning techniques to build predictive models, further improving the accuracy and intelligence of stock analysis.

The planned tasks include:

1. **Data Preparation and Feature Engineering**: Create training datasets.
2. **Model Selection and Training**: Select the appropriate prediction models based on data size and problem nature.
3. **Model Evaluation and Iterative Optimization**: Perform backtesting and cross-validation, evaluating the accuracy and stability of the models.
4. **Result Visualization and Interpretation**: Provide interpretability of the model analysis to help users understand the key factors behind prediction results.

Through modeling analysis, the system will have a certain degree of stock price trend prediction ability, providing more intelligent reference decisions for investors.

### 🚧 Program Features 🚧

(Completion timeline pending)

## 📊 Usage Limits

This platform uses a backend proxy service to call the **FinMind** API to provide real-time stock data. However, this API has a usage limit of **600 calls per hour**, shared by all users.

We apologize for this limitation; currently, we cannot afford more API costs, so only a minimum usage limit is provided. If more resources become available in the future, we will offer higher usage limits.

If you find that data or charts are not displaying correctly, it may be because the API call limit for the current period has been reached. Please try again later or wait for the system to reset the limit in the next hour.

- [**FinMind Official Website**](https://finmindtrade.com/)
- [**FinMind GitHub Documentation**](https://github.com/FinMind/FinMind)

:::info
The maintenance window for the FinMind API is from midnight to 7 AM every Sunday, during which stock data cannot be retrieved.
:::

## 📢 Disclaimer

The following content is for reference only and **does not constitute any investment or financial advice**. Please read carefully and assess the risks.

1. **Non-professional Investment Advice**: The analysis and data provided on this platform are based on historical data and technical indicators, for reference only. The content does not guarantee or predict market trends, nor does it have legal authority.
2. **Investment Risks and Self-responsibility**: Investment involves risk, and market conditions change rapidly. Any investment behavior may lead to asset losses. Investment decisions based on the data or analysis provided on this platform should be made by investors at their own discretion and risk.
3. **Limitations of Data and Information**: The data and analysis results used on this platform may be subject to delays, incompleteness, or errors from the information source. Algorithms are not absolutely reliable. Therefore, if the data or results differ from your expectations, please rely on official information and authoritative sources.
4. **Past Performance Does Not Predict Future Results**: All analyses and data are based on past market performance, and historical results cannot guarantee future outcomes. Market changes are influenced by various factors and carry high uncertainty, and any investment decisions made based on this content involve risk.
5. **Professional Consultation is Recommended**: Before making any investment decisions, it is advisable to consult a legally qualified professional investment advisor, licensed financial expert, or other trusted professionals for more comprehensive and objective advice.

This platform does not make any express or implied guarantees regarding the accuracy, completeness, or timeliness of the content, nor does it bear any direct or indirect liability. Please invest based on your own risk tolerance and make decisions with caution.
