"""
Data loading and feature engineering for the TFT stock model.

Takes the merged per-ticker daily CSV and produces a frame ready for
pytorch-forecasting's TimeSeriesDataSet:
  - keeps the configured columns
  - drops tickers with too little history
  - builds lag features (lag 1..max_lag) per ticker
  - assigns a contiguous time_idx per ticker
"""
import numpy as np
import pandas as pd

from config import DataConfig, ModelConfig


def load_and_clean(cfg: DataConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df = df[cfg.required_columns]

    # Ratio columns can contain inf from division by zero; treat as missing.
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Drop tickers without enough raw history.
    counts = df["Ticker"].value_counts()
    keep = counts[counts >= cfg.min_rows_per_ticker].index
    df = df[df["Ticker"].isin(keep)].copy()
    return df


def add_lag_features(df: pd.DataFrame, cfg: DataConfig) -> tuple[pd.DataFrame, list[str]]:
    lag_cols = []
    lagged = []
    for var in cfg.lag_source_columns:
        for lag in range(1, cfg.max_lag + 1):
            name = f"{var}_lag_{lag}"
            lagged.append(df.groupby("Ticker")[var].shift(lag).rename(name))
            lag_cols.append(name)
    df = pd.concat([df] + lagged, axis=1)
    df = df.dropna(subset=lag_cols)

    # Re-index time per ticker so time_idx is contiguous after dropping rows.
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df["time_idx"] = df.groupby("Ticker").cumcount()
    return df, lag_cols


def drop_short_tickers(df: pd.DataFrame, cfg: DataConfig, mcfg: ModelConfig) -> pd.DataFrame:
    """Remove tickers too short to yield even one encoder+prediction window."""
    min_required = mcfg.encoder_length + cfg.max_lag + mcfg.prediction_length
    sizes = df.groupby("Ticker").size()
    keep = sizes[sizes >= min_required].index
    return df[df["Ticker"].isin(keep)].copy()


def train_valid_split(df: pd.DataFrame, frac: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-ticker temporal split: last 10% of each ticker's timeline is validation."""
    train_parts, valid_parts = [], []
    for _, g in df.groupby("Ticker"):
        cutoff = int(g["time_idx"].max() * frac)
        train_parts.append(g[g["time_idx"] <= cutoff])
        valid_parts.append(g[g["time_idx"] > cutoff])
    return pd.concat(train_parts), pd.concat(valid_parts)


def build_frame(cfg: DataConfig, mcfg: ModelConfig):
    df = load_and_clean(cfg)
    df, lag_cols = add_lag_features(df, cfg)
    df = drop_short_tickers(df, cfg, mcfg)
    return df, lag_cols
