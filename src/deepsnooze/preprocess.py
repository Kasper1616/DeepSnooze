import torch
import mne
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm  # pip install tqdm

# --- CONFIGURATION ---
RAW_DATA_PATH = Path("data/")
PROCESSED_PATH = Path("data_processed/")
TARGET_SFREQ = 128.0
EPOCH_DURATION = 4.0
TARGET_LEN = int(TARGET_SFREQ * EPOCH_DURATION)

LABEL_MAP = {
    "w": 0, "1": 0,  # Wake
    "n": 1, "2": 1,  # NREM
    "r": 2, "3": 2,  # REM
}

def preprocess():
    PROCESSED_PATH.mkdir(exist_ok=True, parents=True)
    
    # Find all EDF files
    edf_files = sorted(list(RAW_DATA_PATH.glob("Cohort*/recordings/*.edf")))
    
    print(f"Found {len(edf_files)} recordings. Starting preprocessing...")

    for edf_file in tqdm(edf_files):
        # Construct paths
        # We look for scoring in the sibling 'scorings' folder
        scoring_path = edf_file.parent.parent / "scorings" / f"{edf_file.stem}.csv"
        
        if not scoring_path.exists():
            continue
            
        # 1. Load Labels
        try:
            df = pd.read_csv(scoring_path, header=None)
            labels = df[1].astype(str).str.strip().map(LABEL_MAP)
            # Drop rows with unknown labels (NaN)
            valid_indices = labels.dropna().index.to_numpy()
            labels = labels.dropna().to_numpy(dtype=np.int64)
        except Exception:
            continue

        if len(labels) == 0:
            continue

        # 2. Load and Resample Signal
        try:
            # Load full recording
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            
            # Resample WHOLE recording at once (Much faster than per-epoch)
            if raw.info['sfreq'] != TARGET_SFREQ:
                raw.resample(TARGET_SFREQ, npad="auto")
            
            # Get data as numpy [Channels, Time]
            data = raw.get_data()
            
        except Exception as e:
            print(f"Error reading {edf_file.name}: {e}")
            continue

        # 3. Cut into Epochs
        # We collect all valid epochs for this file into one list
        file_epochs = []
        file_labels = []

        samples_per_epoch = TARGET_LEN
        
        for i, original_idx in enumerate(valid_indices):
            start = original_idx * samples_per_epoch
            end = start + samples_per_epoch
            
            # Check bounds
            if end > data.shape[1]:
                break
                
            # Extract epoch
            epoch_data = data[:, start:end]
            
            # Verify shape (sometimes resampling causes off-by-one errors)
            if epoch_data.shape[1] == samples_per_epoch:
                file_epochs.append(epoch_data)
                file_labels.append(labels[i])

        if len(file_epochs) > 0:
            # Stack into tensors
            # Shape: [Num_Epochs, Channels, Time]
            X = torch.tensor(np.array(file_epochs), dtype=torch.float32)
            y = torch.tensor(np.array(file_labels), dtype=torch.long)
            
            # Save to disk
            save_name = PROCESSED_PATH / f"{edf_file.stem}.pt"
            torch.save({"X": X, "y": y}, save_name)

if __name__ == "__main__":
    preprocess()