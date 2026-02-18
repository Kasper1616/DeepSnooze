from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split

class SleepyRatDataset(Dataset):
    def __init__(self, processed_path="data/processed", transform=None):
        self.cache = [
            torch.load(f, map_location="cpu", weights_only=True)
            for f in sorted(Path(processed_path).glob("*.pt"))
        ]

        self.index_map = [
            (file_idx, i)
            for file_idx, data_dict in enumerate(self.cache)
            for i in range(len(data_dict["y"]))
        ]
        self.transform = transform  

    @property
    def labels(self):
        return [
            int(self.cache[file_idx]["y"][sample_idx]) # Use int() to extract the value
            for file_idx, sample_idx in self.index_map
        ]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.index_map[idx]
        d = self.cache[file_idx]
        x, y = d["X"][sample_idx], d["y"][sample_idx]
        if self.transform:
            x = self.transform(x)
        return x, y


class SleepDataModule(LightningDataModule):
    def __init__(
        self,
        processed_path="data/processed",
        batch_size=16,
        subset_size=10_000,
        val_split=0.2,
        num_workers=0,
        transform=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.transform = transform

    def setup(self, stage=None):
        full = SleepyRatDataset(self.hparams["processed_path"], transform=self.transform)
        
        indices = np.arange(len(full))
        labels = np.array(full.labels)
        subset_size = self.hparams.get("subset_size")

        if subset_size and subset_size < len(full):
            indices, _, labels, _ = train_test_split(
                indices, labels,
                train_size=subset_size,
                stratify=labels,
                random_state=42
            )

        train_idx, val_idx = train_test_split(
            indices, 
            test_size=self.hparams["val_split"], 
            stratify=labels,
            random_state=42
        )

        self.train_ds = Subset(full, train_idx)
        self.val_ds = Subset(full, val_idx)

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
    