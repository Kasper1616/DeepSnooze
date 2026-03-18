import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_curves(experiment_name: str, log_dir: str = "logs"):
    csv_path = Path(log_dir) / f"{experiment_name}.csv"
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="train_loss")
    if "val_loss" in df.columns:
        axes[0].plot(df["epoch"], df["val_loss"], label="val_loss")
    if "elbo_loss" in df.columns:
        axes[0].plot(df["epoch"], df["elbo_loss"], label="elbo_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss")

    if "val_acc" in df.columns:
        axes[1].plot(df["epoch"], df["val_acc"], label="val_acc")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].set_title("Validation Accuracy")

    plt.suptitle(experiment_name)
    plt.tight_layout()
    return fig
