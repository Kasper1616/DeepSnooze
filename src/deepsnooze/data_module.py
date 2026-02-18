from pathlib import Path

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset


class SleepyRatDataset(Dataset):
    def __init__(self, processed_path="data/processed", transform=None):
        files = sorted(Path(processed_path).glob("*.pt"))
        self.subjects = [f.stem for f in files]
        self.cache = [
            torch.load(f, map_location="cpu", weights_only=True)
            for f in files
        ]

        self.index_map = [
            (file_idx, i)
            for file_idx, data_dict in enumerate(self.cache)
            for i in range(len(data_dict["y"]))
        ]
        self.transform = transform

    def subject_of(self, global_idx):
        file_idx, _ = self.index_map[global_idx]
        return self.subjects[file_idx]

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
        val_subject="A1",
        num_workers=0,
        transform=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.transform = transform

    def setup(self, stage=None):
        full = SleepyRatDataset(self.hparams["processed_path"], transform=self.transform)
        val_subject = self.hparams["val_subject"]
        train_indices = [i for i in range(len(full)) if full.subject_of(i) != val_subject]
        val_indices = [i for i in range(len(full)) if full.subject_of(i) == val_subject]
        self.train_ds = Subset(full, train_indices)
        self.val_ds = Subset(full, val_indices)

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
    