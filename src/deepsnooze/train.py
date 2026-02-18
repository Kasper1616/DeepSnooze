from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from deepsnooze.data_module import SleepDataModule, SleepyRatDataset
from deepsnooze.models.ffnn import DeepSleepFFNN

from deepsnooze.transforms.standardize_signal import StandardizeSignal
from deepsnooze.transforms.spectrogram_tranform import SpectrogramTransform

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch

if __name__ == "__main__":
    # datamodule = SleepDataModule(
    #     processed_path="data/processed", 
    #     batch_size=64, # Strongly suggest 64 for BatchNorm stability
    #     transform=StandardizeSignal()
    # )

    datamodule = SleepDataModule(
        processed_path="data/processed",
        batch_size=64, 
        val_subject="A1",
        transform=StandardizeSignal()
    )

    datamodule.setup(stage="fit")

    full_ds: SleepyRatDataset = datamodule.train_ds.dataset
    all_labels = np.array(full_ds.labels)
    train_labels = all_labels[datamodule.train_ds.indices]
    

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    
    label_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Calculated Class Weights: {label_weights}")

    # model = DeepSleepFFNN(lr=1e-3, label_weights=label_weights)
    model = DeepSleepFFNN(input_size=3*33*17, lr=1e-3, label_weights=label_weights) # Adjust input size for spectrograms

    trainer = Trainer(
        max_epochs=20,
        callbacks=[
            ModelCheckpoint(
                monitor="val_acc", mode="max", save_top_k=1, dirpath="models/"
            ),
            EarlyStopping(monitor="val_loss", patience=5),
        ],
    )

    trainer.fit(model, datamodule=datamodule)
