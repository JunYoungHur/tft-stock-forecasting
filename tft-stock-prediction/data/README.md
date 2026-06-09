# Data

The actual feature files are not committed (size / licensing). This describes
the expected schema so the code can be run with your own data.

## `merged_data_300.csv`
Per-ticker daily rows with at least these columns:

- `Date` (YYYY-MM-DD), `Ticker`, `Sector`
- Price-derived: `Close`, `Volume`, `RSI`, `MACD_hist`, `MA_112`, `OC_ratio`, `HL_diff`, `Real`
- Macro: `sp500_Close`, `sp500_HL_diff`, `sp500_OC_ratio`,
  `treasury_10yr_Close`, `treasury_10yr_HL_diff`, `treasury_10yr_OC_ratio`,
  `treasury_2yr_Close`, `treasury_2yr_HL_diff`, `treasury_2yr_OC_ratio`
- Known-future: `week_of_year`, `Ichimoku_leading_span_a`

## `prices.csv` (for portfolio.py)
Wide table: index = date, one column per asset, values = close prices.
