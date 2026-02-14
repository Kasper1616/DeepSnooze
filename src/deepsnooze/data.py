import torch
import mne
import pandas as pd
import numpy as np
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
        
        # 1. Store individual samples here instead of just file paths
        # Format: (edf_path, epoch_idx, label_int, sfreq)
        self.samples = [] 

        # Label mapping: merge artifacts with base classes
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
        
        print("Initializing dataset...")
        print(f"Found {len(cohort_dirs)} cohort directories.")

        for cohort_dir in cohort_dirs:
            recordings_dir = cohort_dir / "recordings"
            scorings_dir = cohort_dir / "scorings"

            # Match EDF files with CSV files
            edf_files = sorted(recordings_dir.glob("*.edf"))
            
            for edf_file in edf_files:
                csv_file = scorings_dir / f"{edf_file.stem}.csv"
                
                if csv_file.exists():

                    # 1. Get Sampling Rate quickly (without loading data)
                    try:
                        raw_info = mne.io.read_raw_edf(str(edf_file), preload=False, verbose=False).info
                        sfreq = raw_info["sfreq"]
                    except Exception as e:
                        print(f"Skipping corrupted file {edf_file.name}: {e}")
                        continue

                    # 2. Read Labels
                    try:
                        labels_df = pd.read_csv(csv_file, header=None)
                    except Exception as e:
                        print(f"Could not read CSV {csv_file.name}: {e}")
                        continue

                    # 3. Validate and Store Samples
                    for epoch_idx in range(len(labels_df)):
                        # Get label string safely
                        val = labels_df.iloc[epoch_idx, 1]
                        label_str = str(val).strip() # .strip() removes hidden spaces!

                        # Check if valid
                        if label_str in self.label_map:
                            label_int = self.label_map[label_str]
                            
                            # Store exactly what __getitem__ needs
                            self.samples.append((str(edf_file), epoch_idx, label_int, sfreq))
                        else:
                            pass


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. Unpack the pre-validated data
        edf_path, epoch_idx, label, sfreq = self.samples[idx]

        target_sfreq = 128.0  # Standardize everyone to 128Hz
        target_len = int(self.epoch_duration * target_sfreq)  # 4.0 * 128 = 512 samples

        # 2. Calculate offsets
        samples_per_epoch = int(self.epoch_duration * sfreq)
        start_sample = epoch_idx * samples_per_epoch
        end_sample = start_sample + samples_per_epoch

        # 3. Load Data efficiently
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        
        # MNE handles the specific read here
        data = raw.get_data(start=start_sample, stop=end_sample, verbose=False)

        # Handle edge case: If file ends early (incomplete epoch), pad with zeros
        if data.shape[1] < samples_per_epoch:
            pad_width = samples_per_epoch - data.shape[1]
            data = np.pad(data, ((0,0), (0, pad_width)), mode='constant')

        if sfreq != target_sfreq:
            # mne.filter.resample works on numpy arrays
            # It expects shape (n_channels, n_times)
            data = mne.filter.resample(data, down=sfreq/target_sfreq, verbose=False)
            
            # Double check: Resampling might result in 511 or 513 due to rounding
            # Force exact target length
            if data.shape[1] > target_len:
                data = data[:, :target_len]
            elif data.shape[1] < target_len:
                pad_width = target_len - data.shape[1]
                data = np.pad(data, ((0,0), (0, pad_width)), mode='constant')

        # 4. Convert to tensors
        signal_tensor = torch.from_numpy(data).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return signal_tensor, label_tensor


if __name__ == "__main__":
    from collections import Counter
    dataset = SleepyRatDataset(base_path="data/")
    print(f"Dataset size: {len(dataset)} samples")

    # Example: get first sample
    signal, label = dataset[0]
    print(f"Signal shape: {signal.shape}, Label: {label}")

    # Get distribution of labels
    print("Calculating distribution ...")
    all_labels = [sample[2] for sample in dataset.samples]  # Extract label_int from samples
    label_counts = Counter(all_labels)
    print("Label distribution:", label_counts)
 
    
        
