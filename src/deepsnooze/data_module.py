from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset, random_split


class SleepyRatDataset(Dataset):
    def __init__(self, processed_path="data/processed"):
        self.cache = [
            torch.load(f, map_location="cpu", weights_only=True)
            for f in sorted(Path(processed_path).glob("*.pt"))
        ]

        self.index_map = [
            (file_idx, i)
            for file_idx, data_dict in enumerate(self.cache)
            for i in range(len(data_dict["y"]))
        ]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.index_map[idx]
        d = self.cache[file_idx]
        return d["X"][sample_idx], d["y"][sample_idx]


class SleepDataModule(LightningDataModule):
    def __init__(
        self,
        processed_path="data/processed",
        batch_size=16,
        subset_size=10_000,
        val_split=0.2,
        num_workers=0,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        full = SleepyRatDataset(self.hparams["processed_path"])
        n = min(self.hparams["subset_size"], len(full))
        indices = np.random.choice(len(full), size=n, replace=False)
        subset = Subset(full, indices)
        val_size = int(len(subset) * self.hparams["val_split"])
        train_size = len(subset) - val_size
        self.train_ds, self.val_ds = random_split(subset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            num_workers=self.hparams["num_workers"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams["batch_size"],
            shuffle=False,
            num_workers=self.hparams["num_workers"],
        )
