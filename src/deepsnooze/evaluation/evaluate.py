from pathlib import Path

import hydra
import pyro
import torch
import wandb
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from deepsnooze.data import SleepDataModule
from deepsnooze.data.transforms import SpectrogramTransform, StandardizeSignal, ChannelKnockout
from deepsnooze.models.cnn import DeepCNN, SimpleCNN, SleepyCNN
from deepsnooze.models.ffnn import DeepSleepFFNN
from deepsnooze.models.lora import apply_lora
from deepsnooze.tasks import BayesianClassificationTask, StandardClassificationTask

TRANSFORMS = {
    "none": None,
    "spectrogram": SpectrogramTransform,
    "standardize": StandardizeSignal,
}

MODEL_CLASSES = {
    "cnn": SleepyCNN,
    "cnn_simple": SimpleCNN,
    "cnn_deep": DeepCNN,
    "ffnn": DeepSleepFFNN,
}

TASK_CLASSES = {
    "standard": StandardClassificationTask,
    "bayesian": BayesianClassificationTask,
}

NUM_CHANNELS = 3


def build_datamodule(cfg: DictConfig, test_transform=None) -> SleepDataModule:
    transform_cls = TRANSFORMS.get(cfg.data.transform)
    transform = transform_cls() if transform_cls is not None else None
    dm = SleepDataModule(
        processed_path=cfg.data.processed_path,
        batch_size=cfg.training.batch_size,
        val_subject=cfg.data.val_subject,
        test_subject=cfg.data.test_subject,
        transform=transform,
        test_transform=test_transform,
    )
    dm.setup(stage="test")
    return dm


def load_task(cfg: DictConfig, model):
    task_cls = TASK_CLASSES.get(cfg.training.mode)
    if task_cls is None:
        raise ValueError(f"Unknown training mode: {cfg.training.mode!r}")

    if cfg.training.lora:
        apply_lora(
            model,
            rank=cfg.training.rank,
            alpha=cfg.training.alpha,
            use_bayesian=(cfg.training.mode == "bayesian"),
        )

    if cfg.training.mode == "bayesian":
        pyro.clear_param_store()
        task = task_cls(
            model,
            num_classes=cfg.model.num_classes,
            lr=cfg.model.lr,
            svi_lr=cfg.training.svi_lr,
            variational_family=cfg.training.variational_family,
        )
        pyro_path = Path("models") / f"{cfg.experiment_name}_pyro.pt"
        if not pyro_path.exists():
            raise FileNotFoundError(f"Pyro checkpoint not found: {pyro_path}")
        pyro.get_param_store().load(str(pyro_path))
    else:
        ckpt_path = Path("models") / f"{cfg.experiment_name}.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        task = task_cls(
            model,
            num_classes=cfg.model.num_classes,
            lr=cfg.model.lr,
            loss=cfg.training.loss,
            label_smoothing=cfg.training.label_smoothing,
            focal_gamma=cfg.training.focal_gamma,
        )
        ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model_state = {
            k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")
        }
        model.load_state_dict(model_state)

    return task


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(f"Evaluating: {cfg.experiment_name}")

    model_cls = MODEL_CLASSES.get(cfg.model.name)
    if model_cls is None:
        raise ValueError(f"Unknown model: {cfg.model.name!r}")

    model = model_cls(num_classes=cfg.model.num_classes, lr=cfg.model.lr)
    task = load_task(cfg, model)

    logger = WandbLogger(
        project=cfg.wandb.project,
        name=f"{cfg.experiment_name}_knockout_eval",
        group=cfg.wandb.group,
        notes=cfg.wandb.notes,
    )
    trainer = Trainer(logger=logger, enable_checkpointing=False, enable_model_summary=False)

    conditions = [("clean", None)] + [
        (f"ko_ch{c}", ChannelKnockout(channel=c)) for c in range(NUM_CHANNELS)
    ]

    _CLASSES = ["wake", "nrem", "rem"]

    results = {}
    for label, test_transform in conditions:
        dm = build_datamodule(cfg, test_transform=test_transform)
        out = trainer.test(task, datamodule=dm, verbose=False)
        r = out[0]
        results[label] = {
            "acc":         r.get("test_acc", float("nan")),
            "nll":         r.get("test_nll", float("nan")),
            "f1_macro":    r.get("test_f1_macro", float("nan")),
            "f1_weighted": r.get("test_f1_weighted", float("nan")),
            **{f"f1_{cls}": r.get(f"test_f1_{cls}", float("nan")) for cls in _CLASSES},
        }

    # Log condition-labeled metrics so they're distinguishable in WandB
    log_dict = {}
    for label, m in results.items():
        log_dict[f"eval/{label}_acc"]         = m["acc"]
        log_dict[f"eval/{label}_nll"]         = m["nll"]
        log_dict[f"eval/{label}_f1_macro"]    = m["f1_macro"]
        log_dict[f"eval/{label}_f1_weighted"] = m["f1_weighted"]
        for cls in _CLASSES:
            log_dict[f"eval/{label}_f1_{cls}"] = m[f"f1_{cls}"]
    clean = results["clean"]
    for c in range(NUM_CHANNELS):
        ko = results[f"ko_ch{c}"]
        log_dict[f"eval/drop_ch{c}_acc"]         = clean["acc"] - ko["acc"]
        log_dict[f"eval/drop_ch{c}_nll"]         = ko["nll"] - clean["nll"]
        log_dict[f"eval/drop_ch{c}_f1_macro"]    = clean["f1_macro"] - ko["f1_macro"]
        log_dict[f"eval/drop_ch{c}_f1_weighted"] = clean["f1_weighted"] - ko["f1_weighted"]
        for cls in _CLASSES:
            log_dict[f"eval/drop_ch{c}_f1_{cls}"] = clean[f"f1_{cls}"] - ko[f"f1_{cls}"]
    logger.experiment.log(log_dict)

    print("\n--- Knockout Robustness Results ---")
    print(f"{'Condition':<12}  {'acc':>8}  {'nll':>8}  {'f1_mac':>8}  {'f1_wtd':>8}  {'wake':>8}  {'nrem':>8}  {'rem':>8}")
    print("-" * 84)
    for label, m in results.items():
        print(f"{label:<12}  {m['acc']:>8.4f}  {m['nll']:>8.4f}  {m['f1_macro']:>8.4f}  {m['f1_weighted']:>8.4f}  {m['f1_wake']:>8.4f}  {m['f1_nrem']:>8.4f}  {m['f1_rem']:>8.4f}")
    print("\n--- Drop vs Clean ---")
    for c in range(NUM_CHANNELS):
        ko = results[f"ko_ch{c}"]
        print(f"  channel {c}:  acc {clean['acc']-ko['acc']:+.4f}  nll {ko['nll']-clean['nll']:+.4f}  f1_mac {clean['f1_macro']-ko['f1_macro']:+.4f}  f1_wtd {clean['f1_weighted']-ko['f1_weighted']:+.4f}  wake {clean['f1_wake']-ko['f1_wake']:+.4f}  nrem {clean['f1_nrem']-ko['f1_nrem']:+.4f}  rem {clean['f1_rem']-ko['f1_rem']:+.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()
