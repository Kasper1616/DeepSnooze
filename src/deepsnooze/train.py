import torch
import hydra

from pathlib import Path

from deepsnooze.models.ffnn import DeepSleepFFNN
from deepsnooze.models.cnn import SleepyCNN

from deepsnooze.models.lora import apply_lora

from deepsnooze.transforms import StandardizeSignal, SpectrogramTransform


from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig

from deepsnooze.data import SleepDataModule
from deepsnooze.data.transforms import SpectrogramTransform, StandardizeSignal
from deepsnooze.models.cnn import SleepyCNN
from deepsnooze.models.ffnn import DeepSleepFFNN
from deepsnooze.models.lora import apply_lora
from deepsnooze.tasks import BayesianClassificationTask, StandardClassificationTask

TRANSFORMS = {
    "spectrogram": SpectrogramTransform,
    "standardize": StandardizeSignal,
}

MODEL_CLASSES = {
    "cnn": SleepyCNN,
    "ffnn": DeepSleepFFNN,
}

TASK_CLASSES = {
    "standard": StandardClassificationTask,
    "bayesian": BayesianClassificationTask,
}


def build_datamodule(cfg: DictConfig) -> SleepDataModule:
    transform = TRANSFORMS[cfg.data.transform]()
    dm = SleepDataModule(
        processed_path=cfg.data.processed_path,
        batch_size=cfg.training.batch_size,
        val_subject=cfg.data.val_subject,
        test_subject=cfg.data.test_subject,
        transform=transform,
    )
    dm.setup(stage="fit")
    return dm


def build_model(cfg: DictConfig):
    model_cls = MODEL_CLASSES.get(cfg.model.name)
    if model_cls is None:
        raise ValueError(f"Unknown model: {cfg.model.name!r}")
    return model_cls(num_classes=cfg.model.num_classes, lr=cfg.model.lr)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(f"Experiment: {cfg.experiment_name}")

    datamodule = build_datamodule(cfg)
    print(f"Class weights: {datamodule.class_weights}")

    model = build_model(cfg)

    if cfg.training.lora:
        base_path = Path("models") / f"{cfg.model.name}_base.pt"
        model.load_state_dict(torch.load(base_path, weights_only=True))
        apply_lora(
            model,
            rank=cfg.training.rank,
            alpha=cfg.training.alpha,
            use_bayesian=(cfg.training.mode == "bayesian"),
        )

    task_cls = TASK_CLASSES.get(cfg.training.mode)
    if task_cls is None:
        raise ValueError(f"Unknown training mode: {cfg.training.mode!r}")

    Path("models").mkdir(exist_ok=True)
    if cfg.training.mode == "bayesian":
        pyro_path = str(Path("models") / f"{cfg.experiment_name}_pyro.pt")
        task = task_cls(model, num_classes=cfg.model.num_classes, lr=cfg.model.lr,
                        label_weights=datamodule.class_weights, pyro_checkpoint_path=pyro_path)
        callbacks = []
    else:
        base_model_path = None if cfg.training.lora else str(Path("models") / f"{cfg.model.name}_base.pt")
        task = task_cls(model, num_classes=cfg.model.num_classes, lr=cfg.model.lr,
                        label_weights=datamodule.class_weights, base_model_path=base_model_path)
        callbacks = [
            ModelCheckpoint(monitor="val_acc", mode="max", dirpath="models",
                            filename=cfg.experiment_name, save_weights_only=True),
        ]

    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        logger=TensorBoardLogger(save_dir="logs", name=cfg.experiment_name, version=""),
    )
    trainer.fit(task, datamodule=datamodule)


if __name__ == "__main__":
    main()
