"""
Load a trained TFT and produce quantile predictions + plots.

Usage:
    python src/predict.py --data data/merged_data_300.csv --model artifacts/tft_model.pth
"""
import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from config import DataConfig, ModelConfig, Paths
from data import build_frame, train_valid_split
from model import build_datasets, build_model


def plot_quantile_forecasts(decoder_target, q_low, q_med, q_high, out_dir, n=10):
    """Plot the full prediction horizon per sample: median line + [Q0.2, Q0.8] band."""
    os.makedirs(out_dir, exist_ok=True)
    n = min(n, len(decoder_target))
    idx = np.random.choice(len(decoder_target), size=n, replace=False)
    horizon = decoder_target.shape[1]
    steps = np.arange(1, horizon + 1)

    plt.figure(figsize=(14, 8))
    for i, j in enumerate(idx):
        plt.subplot(2, 5, i + 1)
        plt.fill_between(steps, q_low[j], q_high[j], color="tab:blue",
                         alpha=0.2, label="Q0.2-Q0.8")
        plt.plot(steps, q_med[j], "-o", color="tab:blue", markersize=3, label="Q0.5")
        plt.plot(steps, decoder_target[j], "-o", color="black", markersize=3, label="Actual")
        plt.title(f"Sample {i + 1}", fontsize=10)
        plt.xlabel("step ahead", fontsize=8)
        plt.grid(alpha=0.3)
        if i == 0:
            plt.legend(fontsize=7)
    plt.tight_layout()
    path = os.path.join(out_dir, "quantile_forecasts.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    dcfg, mcfg, paths = DataConfig(), ModelConfig(), Paths()
    if args.data:
        dcfg.data_path = args.data
    model_path = args.model or paths.model_file

    df, lag_cols = build_frame(dcfg, mcfg)
    train_df, valid_df = train_valid_split(df)
    train_ds, val_ds = build_datasets(train_df, valid_df, lag_cols, dcfg, mcfg)
    val_loader = val_ds.to_dataloader(train=False, batch_size=mcfg.batch_size, shuffle=False)

    tft = build_model(train_ds, mcfg)
    tft.load_state_dict(torch.load(model_path, map_location="cpu"))
    tft.eval()

    result = tft.predict(val_loader, mode="quantiles", return_x=True)
    preds, x = result.output, result.x

    # Keep (n_samples, horizon) shape so we can plot the full forecast horizon.
    decoder_target = x["decoder_target"].cpu().numpy()
    q_low = preds[..., 0].cpu().numpy()
    q_med = preds[..., 1].cpu().numpy()
    q_high = preds[..., 2].cpu().numpy()

    # Quantile-coverage check: fraction of actuals inside the [Q0.2, Q0.8] band.
    # A well-calibrated 0.2/0.8 interval should cover ~0.6 of the actuals.
    inside = np.mean((decoder_target >= q_low) & (decoder_target <= q_high))
    print(f"Empirical [0.2, 0.8] coverage: {inside:.3f} (target 0.60)")

    plot_quantile_forecasts(decoder_target, q_low, q_med, q_high, paths.figure_dir)


if __name__ == "__main__":
    main()
