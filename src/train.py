"""
Train the TFT stock model end to end.

Usage:
    python src/train.py --data data/merged_data_300.csv --epochs 30

This is a cleaned-up, runnable version of the original exploratory notebook.
Note: this project is an exploration of TFT on noisy financial data, not a
profitable trading system. See README for the honest write-up of what worked
and what didn't.
"""
import argparse
import os

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

from config import DataConfig, ModelConfig, Paths
from data import build_frame, train_valid_split
from model import build_datasets, build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to merged feature CSV")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    dcfg, mcfg, paths = DataConfig(), ModelConfig(), Paths()
    if args.data:
        dcfg.data_path = args.data
    if args.epochs:
        mcfg.max_epochs = args.epochs

    torch.set_float32_matmul_precision("high")
    os.makedirs(paths.model_dir, exist_ok=True)

    # 1. Data
    df, lag_cols = build_frame(dcfg, mcfg)
    train_df, valid_df = train_valid_split(df)
    print(f"Tickers: {df['Ticker'].nunique()} | "
          f"train rows: {len(train_df)} | valid rows: {len(valid_df)}")

    # 2. Datasets / loaders
    train_ds, val_ds = build_datasets(train_df, valid_df, lag_cols, dcfg, mcfg)
    train_loader = train_ds.to_dataloader(
        train=True, batch_size=mcfg.batch_size, shuffle=True, num_workers=4)
    val_loader = val_ds.to_dataloader(
        train=False, batch_size=mcfg.batch_size, shuffle=False, num_workers=4)

    # 3. Model + trainer
    tft = build_model(train_ds, mcfg)
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=mcfg.max_epochs,
        gradient_clip_val=mcfg.gradient_clip_val,
        accumulate_grad_batches=mcfg.accumulate_grad_batches,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            LearningRateMonitor(),
        ],
        logger=False,
    )

    # 4. Train + save
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    torch.save(tft.state_dict(), paths.model_file)
    print(f"Saved model weights to {paths.model_file}")


if __name__ == "__main__":
    main()
