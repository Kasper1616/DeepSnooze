# src/deepsnooze/data/datamodule.py
import numpy as np
import torch
from lightning import LightningDataModule
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset

from deepsnooze.data.dataset import SleepyRatDataset


class SleepDataModule(LightningDataModule):
    def __init__(
        self,
        processed_path="data/processed",
        batch_size=16,
        val_subject="A1",
        test_subject="D6",
        num_workers=7,
        transform=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["transform"])
        self.transform = transform

    def setup(self, stage=None):
        full = SleepyRatDataset(
            self.hparams["processed_path"], transform=self.transform
        )
        val_subject = self.hparams["val_subject"]
        test_subject = self.hparams["test_subject"]
        exclude_subjects = {val_subject, test_subject}
        train_indices = [
            i for i in range(len(full)) if full.subject_of(i) not in exclude_subjects
        ]
        val_indices = [i for i in range(len(full)) if full.subject_of(i) == val_subject]
        test_indices = [
            i for i in range(len(full)) if full.subject_of(i) == test_subject
        ]
        self.train_ds = Subset(full, train_indices)
        self.val_ds = Subset(full, val_indices)
        self.test_ds = Subset(full, test_indices)

        train_labels = np.array(full.labels)[train_indices]
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_labels),
            y=train_labels,
        )
        self.class_weights = torch.tensor(weights, dtype=torch.float32)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            num_workers=self.hparams["num_workers"],
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams["batch_size"],
            shuffle=False,
            num_workers=self.hparams["num_workers"],
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams["batch_size"],
            shuffle=False,
            num_workers=self.hparams["num_workers"],
        )
