---
sidebar_position: 5
---

# Backtesting System Analysis

## PythonTechAnalysis

Once you have downloaded the MasterLink Python module, use the following command to install the backtesting system:

```powershell
pip install .\MasterLink_PythonAPI\Python_tech_analysis\tech_analysis_api_v2-0.0.5-py3-none-win_amd64.whl
```

:::tip
The version of this package we are using is `0.0.5`.
:::

## Official Technical Documentation

- [**MasterLink - Technical Analysis**](https://mlapi.masterlink.com.tw/web_api/service/document/python-analysis)
- [**Official Code Example: example.py**](https://github.com/DocsaidLab/AutoTraderX/blob/main/MasterLink_PythonAPI/Python_tech_analysis/example.py)

## Core Modules

Similarly, after extracting the MasterLink Python module, we directly analyze its core modules. This part of the code surprisingly remains quite straightforward because all functionalities are encapsulated within `.dll` files, and the Python module merely serves as a simple interface.

### TechAnalysis

This segment of the code begins by importing the `TechAnalysisAPI` object encapsulated within the `.dll` file.

Following that, it defines several events that need to be hooked into the `TechAnalysisAPI`. We won't modify or invoke these functions directly, so they are left as is.

- **`Login`**

  This function must be called first to authenticate with the MasterLink API.

- **`GetHisBS_Stock`**

  This function requires specifying a stock and a date, returning the detailed transaction details ("every Tick") for that stock on the specified date.

- **`SubTA` and `UnSubTA`**

  These functions allow users to subscribe or unsubscribe from specific technical indicators.

- **Technical Indicators**

  Through TechAnalysisAPI, this module supports various technical indicators, including:

  - SMA (Simple Moving Average)
  - EMA (Exponential Moving Average)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - KD (Stochastic Oscillator)
  - CDP (Contrary Decision Point)
  - BBands (Bollinger Bands)

### Model

As mentioned earlier, within this module, there are only two `.py` files: `TechAnalysis` and `Model`.

These files define a series of data structures and classes primarily used for stock market technical analysis.

Below is an analysis and explanation of each part of the code:

- **Enum Types**

  The code defines several enum types used to represent different types of technical indicators, time units for candles, and price rise/fall status:

  - `eTA_Type`: Represents various technical analysis indicators such as SMA, Weighted Moving Average (WMA), EMA, etc.
  - `eNK_Kind`: Represents the time range of K-lines, such as daily, 1-minute, etc.
  - `eRaiseFall`: Represents the rise or fall of prices.

- **Data Structures (Data Classes)**

  These data classes provide structures for storing stock trading data:

  - `TKBarRec`: Stores data for K-line charts, including date, product, time series, price, and volume.
  - `TBSRec`: Stores buy and sell records for specific stocks or commodities.
  - Various technical analysis indicator classes (`ta_sma`, `ta_ema`, `ta_wma`, etc.), each containing corresponding K-line data and calculated indicator values.

- **Technical Analysis Indicator Classes**

  These classes represent different technical analysis indicators and combine indicator values with corresponding K-line data. For example:

  - Classes like `ta_sma`, `ta_ema`, and `ta_wma` represent Simple Moving Average, Exponential Moving Average, and Weighted Moving Average, respectively.
  - The `ta_sar` class includes additional information such as stop-loss points and price rise/fall status.
  - Classes like `ta_rsi` and `ta_macd` provide data for Relative Strength Index and Moving Average Convergence Divergence, respectively.
  - `ta_kd`, `ta_cdp`, and `ta_bbands` classes represent Stochastic Oscillator, Contrary Decision Point, and Bollinger Bands indicators, respectively.

- **`k_setting` Class**

  This class is used to configure technical analysis settings, including product ID, time range, types of technical analysis indicators, and start date.

  It serves as the foundation for setting up and configuring calls to the technical analysis API.
