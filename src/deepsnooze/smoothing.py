import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from scipy.stats import mode

# --- IMPORT YOUR MODEL & TRANSFORM ---
from deepsnooze.models.cnn import SleepyCNN
from deepsnooze.data_module import SleepDataModule
from deepsnooze.transforms.spectrogram_tranform import SpectrogramTransform

# --- CONFIGURATION ---
VAL_RAT_INDEX = 'A1'   # ⚠️ MUST MATCH the rat you left out during training!
WINDOW_SIZE = 5     # 5 windows = ~20 seconds. Smooths out blips smaller than this.
# ---------------------

def smooth_predictions(preds, window_size=5):
    """
    Applies a rolling majority vote to smooth predictions.
    removes impossible transitions like NREM -> REM (4s) -> NREM
    """
    n = len(preds)
    smoothed = np.zeros_like(preds)
    half_window = window_size // 2

    print(f"Applying smoothing (Window Size: {window_size})...")
    
    for i in range(n):
        # Define the window boundaries (e.g., i-2 to i+2)
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        
        # Get neighbors
        neighbors = preds[start:end]
        
        # Majority vote: Find the most common class in the window
        # mode() returns (mode_value, count)
        vote = mode(neighbors, keepdims=True)[0][0]
        smoothed[i] = vote
        
    return smoothed

def main():
    # 1. Setup Paths
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "processed"
    
    # Automatically find the newest checkpoint in 'models/' or 'lightning_logs/'
    # If this picks the wrong one, paste the full path manually below.
    try:
        # Check 'models/' first (where your ModelCheckpoint saves)
        model_path = sorted(Path("models").glob("*.ckpt"))[-1]
    except IndexError:
        # Fallback to 'lightning_logs/'
        model_path = sorted(Path("lightning_logs").glob("**/*.ckpt"))[-1]
        
    print(f"\nLOADING CHECKPOINT: {model_path}")

    # 2. Load the Trained Model
    # We load it onto the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SleepyCNN.load_from_checkpoint(model_path)
    model.to(device)
    model.eval()

    # 3. Load the Validation Data (The rat the model has never seen)
    print(f"Loading Validation Data (Rat Index {VAL_RAT_INDEX})...")
    datamodule = SleepDataModule(
        processed_path=str(data_path), 
        val_subject=VAL_RAT_INDEX, 
        batch_size=64, 
        transform=SpectrogramTransform(n_fft=64, hop_length=32),
        num_workers=0 # Set to 4 if on Linux/Mac
    )
    datamodule.setup()
    val_loader = datamodule.val_dataloader()
    
    # 4. Run Inference (Generate Raw Predictions)
    print("Generating Raw Predictions...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            x, y = batch
            x = x.to(device)
            
            # Forward pass
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 5. Apply Smoothing
    smoothed_preds = smooth_predictions(all_preds, window_size=WINDOW_SIZE)

    # 6. Print Reports
    target_names = ['Wake', 'NREM', 'REM']
    
    print("\n" + "="*30)
    print(" 1. RAW MODEL OUTPUT (No Smoothing)")
    print("="*30)
    print(classification_report(all_targets, all_preds, target_names=target_names, digits=4))
    
    print("\n" + "="*30)
    print(f" 2. SMOOTHED OUTPUT (Window={WINDOW_SIZE})")
    print("="*30)
    print(classification_report(all_targets, smoothed_preds, target_names=target_names, digits=4))

    # Optional: Print Confusion Matrix for the Smoothed version
    print("\nConfusion Matrix (Smoothed):")
    print(confusion_matrix(all_targets, smoothed_preds))

if __name__ == "__main__":
    main()