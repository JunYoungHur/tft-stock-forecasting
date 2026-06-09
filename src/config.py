"""Configuration: paths and hyperparameters for the TFT stock model."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    # Path to the merged feature CSV (per-ticker daily rows).
    data_path: str = "data/merged_data_300.csv"
    # Tickers with fewer than this many rows are dropped (not enough history).
    min_rows_per_ticker: int = 380
    # Columns kept from the raw merged file.
    required_columns: List[str] = field(default_factory=lambda: [
        "Date", "Ticker", "Real",
        # static
        "Sector",
        # per-stock price-derived features
        "OC_ratio", "HL_diff",
        # past observed inputs
        "Close", "Volume", "RSI", "MACD_hist", "MA_112",
        # macro context
        "sp500_Close", "sp500_HL_diff", "sp500_OC_ratio",
        "treasury_10yr_Close", "treasury_10yr_HL_diff", "treasury_10yr_OC_ratio",
        "treasury_2yr_Close", "treasury_2yr_HL_diff", "treasury_2yr_OC_ratio",
        # known-future inputs
        "Ichimoku_leading_span_a", "week_of_year",
    ])
    # Variables that get lag features (lag 1..5).
    lag_source_columns: List[str] = field(default_factory=lambda: [
        "OC_ratio", "HL_diff", "Volume", "RSI", "Close", "Real",
        "MACD_hist", "MA_112",
        "sp500_Close", "sp500_HL_diff", "sp500_OC_ratio",
        "treasury_10yr_Close", "treasury_10yr_HL_diff", "treasury_10yr_OC_ratio",
        "treasury_2yr_Close", "treasury_2yr_HL_diff", "treasury_2yr_OC_ratio",
    ])
    max_lag: int = 5


@dataclass
class ModelConfig:
    encoder_length: int = 112      # ~ half a trading year of context
    prediction_length: int = 5     # predict one trading week ahead
    hidden_size: int = 128
    attention_head_size: int = 4
    dropout: float = 0.1
    hidden_continuous_size: int = 32
    learning_rate: float = 0.01
    weight_decay: float = 1e-5
    quantiles: List[float] = field(default_factory=lambda: [0.2, 0.5, 0.8])
    batch_size: int = 128
    max_epochs: int = 30
    gradient_clip_val: float = 0.3
    accumulate_grad_batches: int = 8


@dataclass
class Paths:
    model_dir: str = "artifacts"
    model_file: str = "artifacts/tft_model.pth"
    figure_dir: str = "artifacts/figures"
