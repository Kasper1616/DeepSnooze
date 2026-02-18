from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from deepsnooze.data_module import SleepDataModule
from deepsnooze.models.ffnn import SimpleFFNN

if __name__ == "__main__":
    model = SimpleFFNN(lr=1e-3)
    datamodule = SleepDataModule(processed_path="data/processed", batch_size=16)

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
