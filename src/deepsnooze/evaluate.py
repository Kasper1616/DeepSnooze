import torch
from lightning import Trainer

from deepsnooze.data_module import SleepDataModule, SleepyRatDataset
from deepsnooze.models.cnn import SleepyCNN
from deepsnooze.models.ffnn import DeepSleepFFNN
from deepsnooze.transforms.spectrogram_tranform import SpectrogramTransform
from deepsnooze.transforms.standardize_signal import StandardizeSignal

torch.serialization.add_safe_globals([StandardizeSignal, SpectrogramTransform])


if __name__ == "__main__":
    # datamodule = SleepDataModule(
    #     processed_path="data/processed",
    #     batch_size=128,
    #     val_subject="A1",
    #     test_subject="D6",
    #     transform=StandardizeSignal(),
    # )

    datamodule = SleepDataModule(
        processed_path="data/processed",
        batch_size=128,
        val_subject="B4",
        test_subject="C1",
        eval_transform=SpectrogramTransform(),
        num_workers=4,
    )

    # 1. Load the Trained Model
    # We load it onto the GPU if available
    # model = DeepSleepFFNN.load_from_checkpoint("models/latest.ckpt")
    model = SleepyCNN.load_from_checkpoint("models/epoch=4-step=16875.ckpt", weights_only=False)

    # 2. Run Test Set Evaluation
    trainer = Trainer(logger=False)
    trainer.test(model, datamodule=datamodule)
