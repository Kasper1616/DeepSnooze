from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from deepsnooze.data_module import SleepDataModule, SleepyRatDataset
from deepsnooze.models.ffnn import DeepSleepFFNN
from deepsnooze.models.cnn import SleepyCNN

from deepsnooze.transforms import StandardizeSignal, SpectrogramTransform
from deepsnooze.lora import apply_lora

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch


def train(model="cnn", max_epochs=100, batch_size=32, lr=1e-3, lora=False):
    datamodule = SleepDataModule(
        processed_path="data/processed",
        batch_size=batch_size,
        val_subject="A1",
        test_subject="C2",
        transform=SpectrogramTransform(),
    )

    datamodule.setup(stage="fit")

    full_ds: SleepyRatDataset = datamodule.train_ds.dataset
    all_labels = np.array(full_ds.labels)
    train_labels = all_labels[datamodule.train_ds.indices]

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(train_labels), y=train_labels
    )

    label_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Calculated Class Weights: {label_weights}")

    if lora:
        print("Using LoRA for fine-tuning.")

        # load the pre-trained model
        if model == "ffnn":
            base_model = DeepSleepFFNN(lr=lr, label_weights=label_weights)
        else:
            base_model = SleepyCNN(lr=lr, label_weights=label_weights)
        checkpoint = torch.load(f"models/{base_model.__class__.__name__}.ckpt")

        base_model.load_state_dict(checkpoint["state_dict"], strict=False)

        # apply_lora
        apply_lora(base_model, rank=1, alpha=10, use_bayesian=False)

        model = base_model

    else:
        print("Training the full model from scratch.")
        if model == "ffnn":
            model = DeepSleepFFNN(lr=lr, label_weights=label_weights)
        else:
            model = SleepyCNN(lr=lr, label_weights=label_weights)

    trainer = Trainer(
        max_epochs=max_epochs,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.save_checkpoint(f"models/{model.__class__.__name__}.ckpt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a sleep stage classification model."
    )
    parser.add_argument("--lora", action="store_true", help="Use LoRA for fine-tuning.")

    args = parser.parse_args()
    train(lora=args.lora)
