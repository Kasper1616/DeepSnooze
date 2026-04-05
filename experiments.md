# DeepSnooze Experiments

All runs are tracked in the W&B project **deepsnooze**, grouped by experiment.

## Data

- **Task**: 3-class sleep staging — Wake, NREM, REM
- **Input**: 3-channel EEG/EMG signals (512 samples) → log-scaled spectrogram (3 × 33 × 17)
- **Subjects**: 22 rats, ~475k samples total
- **Split**: leave-one-subject-out (val: A1, test: D6)
- **Class imbalance**: handled via balanced class weights in all experiments

---

## Experiment 1 — Architecture comparison (`baseline`)

**W&B group**: `baseline`  
**Job**: `bash jobs/baseline.sh`

Compares three CNN architectures with standard cross-entropy loss to find the best base model for subsequent LoRA fine-tuning.

| Run | Model | Conv blocks | Channels | FC layers |
|-----|-------|-------------|----------|-----------|
| `cnn_simple_standard` | SimpleCNN | 2 | 16→32 | 128→3 |
| `cnn_standard` | SleepyCNN | 3 | 32→64→128 | 1024→256→3 |
| `cnn_deep_standard` | DeepCNN | 4 | 32→64→128→256 | 512→256→3 |

---

## Experiment 2 — Focal loss (`focal_loss`)

**W&B group**: `focal_loss`  
**Job**: `bash jobs/focal_loss.sh`

Replaces cross-entropy with focal loss (γ=2.0) across all 3 architectures. Focal loss down-weights easy examples (Wake, NREM) and focuses training on hard ones (REM), addressing class imbalance beyond what balanced weights alone provide.

---

## Experiment 3 — Label smoothing (`label_smoothing`)

**W&B group**: `label_smoothing`  
**Job**: `bash jobs/label_smoothing.sh`

Adds label smoothing (α=0.1) to cross-entropy across all 3 architectures. Softens hard targets from 1.0 to 0.9, reducing overconfidence on majority classes and improving calibration (ECE).

---

## Experiment 4 — LoRA fine-tuning (`lora`)

**W&B group**: `lora`  
**Job**: `bash jobs/lora.sh`  
**Prerequisite**: Experiment 1 must complete first — requires `models/cnn_deep_base.pt`

The base `cnn_deep` model is frozen after baseline training. Only the LoRA layers added to the FC linear layers are trained. This experiment compares standard LoRA vs Bayesian LoRA, and tests whether rank (expressivity of the LoRA update) matters for this task.

| Run | Type | Rank | Trainable params | Notes |
|-----|------|------|-----------------|-------|
| `cnn_deep_lora_r1` | Standard LoRA | 1 | ~minimal | Minimal adaptation, deterministic |
| `cnn_deep_lora_r5` | Standard LoRA | 5 | ~5× more | More expressive, deterministic |
| `cnn_deep_bayesian_lora_r1` | Bayesian LoRA | 1 | ~minimal | Uncertainty via Pyro SVI, ELBO loss |
| `cnn_deep_bayesian_lora_r5` | Bayesian LoRA | 5 | ~5× more | Most expressive, uncertainty-aware |

**Standard LoRA**: The LoRA matrices A and B are learned point estimates. The final layer output is `W·x + α·(A·B)·x`.

**Bayesian LoRA**: A and B are treated as random variables with learned posteriors (via `AutoNormal` guide). At inference, a sample from the posterior is drawn each forward pass, giving uncertainty estimates. Trained via ELBO (evidence lower bound) rather than cross-entropy directly.

**Key question**: Does Bayesian LoRA improve calibration (lower ECE, lower NLL) compared to standard LoRA, and does rank matter?

---

## Key metrics to compare across experiments

| Metric | What it measures |
|--------|-----------------|
| `val_acc` | Overall accuracy |
| `val_f1_rem` | REM detection — primary target |
| `val_f1_nrem` | NREM detection |
| `val_f1_wake` | Wake detection |
| `val_ece` | Calibration quality |
| `val_nll` | Probabilistic quality |
| `val_brier` | Proper scoring rule |
