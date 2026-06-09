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


def plot_quantile_samples(decoder_target, q_low, q_med, q_high, out_dir, n=10):
    os.makedirs(out_dir, exist_ok=True)
    idx = np.random.choice(len(decoder_target), size=min(n, len(decoder_target)), replace=False)
    plt.figure(figsize=(14, 8))
    for i, j in enumerate(idx):
        plt.subplot(2, 5, i + 1)
        plt.plot([0], [decoder_target[j]], "go", label="Actual", markersize=8)
        plt.plot([0], [q_low[j]], "rx", label="Q0.2", markersize=8)
        plt.plot([0], [q_med[j]], "yo", label="Q0.5", markersize=8)
        plt.plot([0], [q_high[j]], "m^", label="Q0.8", markersize=8)
        plt.title(f"Sample {i + 1}", fontsize=10)
        plt.xticks([])
        plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "quantile_samples.png")
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

    decoder_target = x["decoder_target"].cpu().numpy().flatten()
    q_low = preds[..., 0].cpu().numpy().flatten()
    q_med = preds[..., 1].cpu().numpy().flatten()
    q_high = preds[..., 2].cpu().numpy().flatten()

    # Simple quantile-coverage sanity check: fraction of actuals inside [Q0.2, Q0.8].
    inside = np.mean((decoder_target >= q_low) & (decoder_target <= q_high))
    print(f"Empirical coverage of the [0.2, 0.8] interval: {inside:.3f} "
          f"(ideal ~0.60)")

    plot_quantile_samples(decoder_target, q_low, q_med, q_high, paths.figure_dir)


if __name__ == "__main__":
    main()
