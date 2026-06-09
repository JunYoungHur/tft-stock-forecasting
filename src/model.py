"""
TimeSeriesDataSet + TemporalFusionTransformer construction.

Mirrors the original experiment's design:
  - static categorical: Sector
  - known-future reals: week_of_year, Ichimoku_leading_span_a
  - unknown reals: the price/macro features plus their lags
  - target: Close, normalized per ticker (GroupNormalizer)
  - QuantileLoss for probabilistic (0.2/0.5/0.8) output
"""
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from config import DataConfig, ModelConfig


def make_validation_windows(valid_df: pd.DataFrame, mcfg: ModelConfig) -> pd.DataFrame:
    """Take the final encoder+prediction window from each ticker for validation."""
    span = mcfg.encoder_length + mcfg.prediction_length
    windows = []
    for ticker, g in valid_df.groupby("Ticker"):
        g = g.sort_values("time_idx").reset_index(drop=True)
        if len(g) >= span:
            windows.append(g.iloc[-span:].copy())
    if not windows:
        raise ValueError("No validation windows could be created.")
    return pd.concat(windows).reset_index(drop=True)


def build_datasets(train_df, valid_df, lag_cols, cfg: DataConfig, mcfg: ModelConfig):
    unknown_reals = cfg.lag_source_columns + lag_cols
    common = dict(
        time_idx="time_idx",
        target="Close",
        group_ids=["Ticker"],
        static_categoricals=["Sector"],
        time_varying_known_reals=["week_of_year", "Ichimoku_leading_span_a"],
        time_varying_unknown_reals=unknown_reals,
        max_encoder_length=mcfg.encoder_length,
        min_encoder_length=mcfg.encoder_length,
        max_prediction_length=mcfg.prediction_length,
        min_prediction_length=mcfg.prediction_length,
        target_normalizer=GroupNormalizer(groups=["Ticker"]),
        allow_missing_timesteps=True,
    )
    train_ds = TimeSeriesDataSet(train_df, **common)
    val_windows = make_validation_windows(valid_df, mcfg)
    val_ds = TimeSeriesDataSet(val_windows, **common)
    return train_ds, val_ds


def build_model(train_ds: TimeSeriesDataSet, mcfg: ModelConfig) -> TemporalFusionTransformer:
    return TemporalFusionTransformer.from_dataset(
        train_ds,
        optimizer="adam",
        learning_rate=mcfg.learning_rate,
        hidden_size=mcfg.hidden_size,
        attention_head_size=mcfg.attention_head_size,
        dropout=mcfg.dropout,
        hidden_continuous_size=mcfg.hidden_continuous_size,
        output_size=len(mcfg.quantiles),
        loss=QuantileLoss(quantiles=mcfg.quantiles),
        reduce_on_plateau_patience=3,
        weight_decay=mcfg.weight_decay,
    )
