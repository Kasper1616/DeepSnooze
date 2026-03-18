# src/deepsnooze/data/dataset.py
from pathlib import Path

import torch
from torch.utils.data import Dataset


class SleepyRatDataset(Dataset):
    def __init__(self, processed_path="data/processed", transform=None):
        files = sorted(Path(processed_path).glob("*.pt"))
        self.subjects = [f.stem for f in files]
        self.cache = [
            torch.load(f, map_location="cpu", weights_only=True) for f in files
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
            int(self.cache[file_idx]["y"][sample_idx])
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
