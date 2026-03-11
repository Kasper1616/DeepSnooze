from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from deepsnooze.data_module import SleepDataModule, SleepyRatDataset
from deepsnooze.models.ffnn import DeepSleepFFNN
from deepsnooze.models.cnn import SleepyCNN

from deepsnooze.transforms import StandardizeSignal, SpectrogramTransform, SpecAugment
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch

if __name__ == "__main__":
    # datamodule = SleepDataModule(
    #     processed_path="data/processed", 
    #     batch_size=64, # Strongly suggest 64 for BatchNorm stability
    #     transform=StandardizeSignal()
    # )

    # 1. Training Transform: Generates spectrograms AND adds synthetic masks
    train_transform = transforms.Compose([
        SpectrogramTransform(n_fft=64, hop_length=32),
        SpecAugment(freq_mask_param=5, time_mask_param=4)
    ])

    # 2. Validation/Test Transform: Generates spectrograms, but NO masking
    eval_transform = SpectrogramTransform(n_fft=64, hop_length=32)

    datamodule = SleepDataModule(
        processed_path="data/processed",
        batch_size=128, 
        val_subject="B4",
        test_subject="C1",
        train_transform=train_transform,
        eval_transform=eval_transform,
        num_workers=4,
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
    model = SleepyCNN(lr=5e-4, label_weights=label_weights) # Adjust input size for spectrograms

    trainer = Trainer(
        max_epochs=50,
        callbacks=[
            ModelCheckpoint(
                monitor="val_acc", mode="max", save_top_k=1, dirpath="models/"
            ),
            # EarlyStopping(monitor="val_loss", patience=50),
        ],
    )

    trainer.fit(model, datamodule=datamodule)
