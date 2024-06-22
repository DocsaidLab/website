---
sidebar_position: 8
---

# Backtesting System

After analyzing the MasterLink's backtesting system Python API, we can now develop a customized backtesting system tailored to our needs.

## Account Login

You can directly embed your account credentials within the class inputs, or alternatively, follow our approach: using a YAML file to store account information.

In the parameter file, ensure to include both the username and password, essential for successful login to your MasterLink account.

Next, import the `BackTesting` class from `autotraderx`:

```python
from autotraderx import load_yaml
from autotraderx.masterlink import BackTesting

# Load account info
cfg = load_yaml(DIR / "account.yaml")

# Login to account
handler = BackTesting(
    user=cfg["user"],
    password=cfg["password"],
)
```

## Subscribing to Indicators

While testing the provided official example code, we encountered a prolonged period of non-responsiveness when subscribing to indicators.

Here's the example code provided by the official documentation:

```python
ta = TechAnalysis(...)

opt = input("1: Indicator\n2: Historical Trades\n> ")
if opt == "1":
    k_config = option()
    ta.SubTA(k_config)
    input("running...\n")
    ta.UnSubTA(k_config)
```

Upon execution, we faced a delay stuck at "running..." for approximately five minutes, leading us to terminate the program due to impatience.

Moreover, since most technical indicators are derived from price-volume data, we find it more efficient to retrieve price-volume data ourselves and compute the indicators.

Therefore, we won't be implementing this part of the functionality.

## Historical Trades

Once logged into the account, you can use our encapsulated `get_data` function to retrieve data:

```python
data = handler.get_data(
    prod_id="2330",
    date="20240102",
)
```

Here, `prod_id` represents the stock code, and `date` follows the format `YYYYMMDD`.

Upon executing the above code, it returns a structure in the form of List[Dict]:

```json
[
    ...omitting above data...
    {
        "成交價格": 590.0,
        "成交時間": 132459.132661,
        "成交量": 1,
        "股票代號": "2330",
        "試搓": False,
        "買賣": 2
    },
    {
        "成交價格": 593.0,
        "成交時間": 133000.0,
        "成交量": 3704,
        "股票代號": "2330",
        "試搓": False,
        "買賣": 2
    }
]
```

Based on our observation, we discovered several features in the MasterLink API:

1. **`試搓` (Test Rounding)**: Indicates whether the transaction is a test rounding.
2. **`買賣` (Buy/Sell)**: Represents whether the transaction occurred in the bid or ask, where `0` signifies a neutral position, `1` represents the bid, and `2` indicates the ask.
3. **`成交時間` (Transaction Time)**: This field is read as `HHMMSS.ffffff`, where `HH` is hours, `MM` is minutes, `SS` is seconds, and `ffffff` represents microseconds.
4. We attempted to query historical data and found records available only from mid-April 2022 onwards.

However, for our modest-scale usage, having data spanning over two years should suffice.

## Next Steps

Our plan is to first retrieve available stock transaction data and establish our own database system.

For subsequent tasks such as calculating technical indicators or backtesting trading strategies, we can directly access data from our database, ensuring speed and convenience.

As for acquiring data from earlier periods, we'll address those needs as they arise.
