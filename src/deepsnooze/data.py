import torch
import mne
import pandas as pd
from pathlib import Path

from torch.utils.data import Dataset


class SleepyRatDataset(Dataset):
    def __init__(self, base_path):
        """
        Args:
            base_path: Base directory containing Cohort* folders
        """
        self.base_path = Path(base_path)
        self.epoch_duration = 4.0  # seconds
        self.data_paths = []
        self.label_paths = []
        self.epoch_indices = []  # (file_idx, epoch_idx) for each sample

        # Label mapping: merge artifacts with base classes
        # TODO: Verify these mappings with Michael
        self.label_map = {
            "w": 0, # Wake
            "1": 0, # Wake artifact
            "n": 1, # NREM
            "2": 1, # NREM artifact
            "r": 2, # REM
            "3": 2, # REM artifact
        }

        # Find all cohort directories
        cohort_dirs = sorted(self.base_path.glob("Cohort*"))

        for cohort_dir in cohort_dirs:
            # Get all EDF files in recordings/
            recordings_dir = cohort_dir / "recordings"
            scorings_dir = cohort_dir / "scorings"

            # Match EDF files with CSV files
            edf_files = sorted(recordings_dir.glob("*.edf"))
            for edf_file in edf_files:
                # Find corresponding CSV (assuming same filename)
                csv_file = scorings_dir / f"{edf_file.stem}.csv"
                if csv_file.exists():
                    file_idx = len(self.data_paths)
                    self.data_paths.append(edf_file)
                    self.label_paths.append(csv_file)

                    # Count epochs in this file
                    labels_df = pd.read_csv(csv_file)
                    n_epochs = len(labels_df)

                    # Add (file_idx, epoch_idx) for each epoch
                    for epoch_idx in range(n_epochs):
                        self.epoch_indices.append((file_idx, epoch_idx))

    def __len__(self):
        return len(self.epoch_indices)

    def __getitem__(self, idx):
        file_idx, epoch_idx = self.epoch_indices[idx]

        # Load EDF file
        raw = mne.io.read_raw_edf(
            str(self.data_paths[file_idx]), preload=True, verbose=False
        )
        sfreq = raw.info["sfreq"]

        # Calculate samples per epoch
        samples_per_epoch = int(self.epoch_duration * sfreq)
        start_sample = epoch_idx * samples_per_epoch
        end_sample = start_sample + samples_per_epoch

        # Extract epoch data (3 channels: EEG1, EEG2, EMG)
        data = raw.get_data(
            start=start_sample, stop=end_sample
        )  # Shape: (3, samples_per_epoch)

        # Load label for this epoch (first expert, first column)
        labels_df = pd.read_csv(self.label_paths[file_idx])
        label_str = str(labels_df.iloc[epoch_idx, 0])
        label = self.label_map.get(label_str) # Map to 0, 1, 2 (Wake, NREM, REM)

        # Convert to tensors
        signal_tensor = torch.from_numpy(data).float()
        label_tensor = torch.tensor(label)

        return signal_tensor, label_tensor


if __name__ == "__main__":
    dataset = SleepyRatDataset(base_path="data/")
    print(f"Dataset size: {len(dataset)} samples")

    # Example: get first sample
    signal, label = dataset[0]
    print(f"Signal shape: {signal.shape}, Label: {label}")
