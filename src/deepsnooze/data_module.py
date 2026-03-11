from pathlib import Path

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset

class TransformSubset(Dataset):
    """A simple wrapper to apply different transforms to PyTorch Subsets."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
    
    @property
    def dataset(self):
        """Exposes the original dataset so train.py can access it."""
        return self.subset.dataset
    
    @property
    def indices(self):
        """Exposes the subset indices so train.py can slice labels."""
        return self.subset.indices


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
        test_subject="D6",
        num_workers=0,
        train_transform=None,  # Split into train and eval transforms
        eval_transform=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_transform = train_transform
        self.eval_transform = eval_transform

    def setup(self, stage=None):
        # 1. Base dataset has NO transforms
        full = SleepyRatDataset(self.hparams["processed_path"], transform=None)
        
        val_subject = self.hparams["val_subject"]
        test_subject = self.hparams["test_subject"]
        exclude_subjects = {val_subject, test_subject}
        
        train_indices = [i for i in range(len(full)) if full.subject_of(i) not in exclude_subjects]
        val_indices = [i for i in range(len(full)) if full.subject_of(i) == val_subject]
        test_indices = [i for i in range(len(full)) if full.subject_of(i) == test_subject]
        
        # 2. Wrap the Subsets with their respective transforms
        self.train_ds = TransformSubset(Subset(full, train_indices), transform=self.train_transform)
        self.val_ds = TransformSubset(Subset(full, val_indices), transform=self.eval_transform)
        self.test_ds = TransformSubset(Subset(full, test_indices), transform=self.eval_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            num_workers=self.hparams["num_workers"],
            persistent_workers=True if self.hparams["num_workers"] > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams["batch_size"],
            shuffle=False,
            num_workers=self.hparams["num_workers"],
            persistent_workers=True if self.hparams["num_workers"] > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams["batch_size"],
            shuffle=False,
            num_workers=self.hparams["num_workers"],
        )
