from pathlib import Path

import hydra
import pyro
import torch
import wandb  # Added to properly close runs in the loop
import numpy as np # Added to calculate mean/std of metrics
import gc
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from deepsnooze.data import SleepDataModule
from deepsnooze.data.transforms import SpectrogramTransform, StandardizeSignal, Knockout
from deepsnooze.models.cnn import DeepCNN, SimpleCNN, SleepyCNN
from deepsnooze.models.ffnn import DeepSleepFFNN
from deepsnooze.models.lora import apply_lora
from deepsnooze.tasks import BayesianClassificationTask, StandardClassificationTask



TRANSFORMS = {
    "none": None,
    "spectrogram": SpectrogramTransform,
    "standardize": StandardizeSignal,
    "knockout": Knockout,
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


def build_datamodule(cfg: DictConfig) -> SleepDataModule:
    transform_cls = TRANSFORMS.get(cfg.data.transform)
    
    transform = transform_cls() if transform_cls is not None else None

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


def run_fold(cfg: DictConfig, fold_idx: int, subjects: str) -> float:
     # 1. Rotate the Test and Validation subjects
    test_subject = subjects[fold_idx]
    # Pick the next subject in the list to act as the validation subject
    val_subject = subjects[(fold_idx + 1) % len(subjects)]
    
    print(f"\n--- Starting Fold {fold_idx + 1}/{len(subjects)} ---")
    print(f"Training on all subjects EXCEPT -> Val: {val_subject} | Test: {test_subject}")
    
    # Update the config for this specific fold
    cfg.data.test_subject = test_subject
    cfg.data.val_subject = val_subject
    
    # Create a unique name for this fold
    fold_exp_name = f"{cfg.experiment_name}_val_{val_subject}_test_{test_subject}"

    datamodule = build_datamodule(cfg)
    model = build_model(cfg)
    if cfg.training.lora:
        base_experiment = cfg.training.get("base_experiment")
        if base_experiment:
            base_ckpt_path = Path("models") / f"{base_experiment}_val_{val_subject}_test_{test_subject}.ckpt"
        else:
            base_ckpt_path = Path("models") / f"{cfg.model.name}_base.pt"
        if not base_ckpt_path.exists():
            raise FileNotFoundError(f"Base checkpoint not found: {base_ckpt_path}")
        ckpt = torch.load(base_ckpt_path, weights_only=True, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model_state = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
        model.load_state_dict(model_state)
        apply_lora(
            model,
            rank=cfg.training.rank,
            alpha=cfg.training.alpha,
            use_bayesian=(cfg.training.mode == "bayesian"),
            freeze_a=cfg.training.get("freeze_a", False),
        )

    task_cls = TASK_CLASSES.get(cfg.training.mode)
    if task_cls is None:
        raise ValueError(f"Unknown training mode: {cfg.training.mode!r}")

    Path("models").mkdir(exist_ok=True)
    if cfg.training.mode == "bayesian":
        pyro_path = str(Path("models") / f"{fold_exp_name}_pyro.pt")
        task = task_cls(model, num_classes=cfg.model.num_classes, lr=cfg.model.lr,
                        label_weights=datamodule.class_weights, pyro_checkpoint_path=pyro_path,
                        svi_lr=cfg.training.svi_lr, variational_family=cfg.training.variational_family)
        callbacks = []
    else:
        base_model_path = None if cfg.training.lora else str(Path("models") / f"{cfg.model.name}_base.pt")
        task = task_cls(model, num_classes=cfg.model.num_classes, lr=cfg.model.lr,
                        label_weights=datamodule.class_weights, base_model_path=base_model_path,
                        loss=cfg.training.loss, label_smoothing=cfg.training.label_smoothing,
                        focal_gamma=cfg.training.focal_gamma)
        callbacks = [
            ModelCheckpoint(
                monitor="val_acc", 
                mode="max", 
                dirpath="models",
                filename=fold_exp_name, 
                save_weights_only=True
            ),
        ]

    logger = WandbLogger(
        project=cfg.wandb.project, 
        name=fold_exp_name,
        group=cfg.wandb.group,
        notes=cfg.wandb.notes
    )

    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        logger=logger,
        enable_model_summary=(fold_idx == 0),
    )
    
    # 2. Fit the model (uses train and val sets)
    trainer.fit(task, datamodule=datamodule)
    
    # 3. Test the model on the held-out subject!
    # Use ckpt_path="best" to load the weights from the highest val_acc
    # Note: If Bayesian mode doesn't use ModelCheckpoint, set ckpt_path=None
    if cfg.training.mode == "bayesian":
        state = torch.load(pyro_path, map_location="cpu", weights_only=False)
        pyro.get_param_store().set_state(state)
        ckpt = None
    else:
        ckpt = "best"
    print(f"\n--- Evaluating Held-Out Test Subject: {test_subject} ---")
    test_results = trainer.test(task, datamodule=datamodule, ckpt_path=ckpt)
    
    # 4. Extract the test metric (Ensure your Task logs "test_acc" in test_step)
    # trainer.test returns a list of dictionaries (one per dataloader)
    test_acc = test_results[0].get("test_acc", 0.0)
    
    wandb.finish()

    # Clean up to free memory before the next fold
    del trainer, model, datamodule, task
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return test_acc


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(f"Base Experiment Name: {cfg.experiment_name}")

    if "cv_subjects" in cfg.data and cfg.data.cv_subjects:
        subjects = cfg.data.cv_subjects
        print(f"Running Cross-Validation over {len(subjects)} subjects.")
        
        fold_metrics = []
        # Pass the whole list to the function so it can rotate val and test
        for i in range(len(subjects)):
            OmegaConf.set_struct(cfg, False)
            fold_acc = run_fold(cfg, fold_idx=i, subjects=subjects)
            fold_metrics.append(fold_acc)
            OmegaConf.set_struct(cfg, True)
            
            print(f"--- Test Accuracy for Held-out Subject {subjects[i]}: {fold_acc:.4f} ---")
            
        mean_acc = np.mean(fold_metrics)
        std_acc = np.std(fold_metrics)
        print("\n=========================================")
        print(f"Cross-Validation Complete!")
        print(f"Average Generalization (Test) Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print("=========================================")
        
    else:
        # Fallback to single run using the subjects defined in the base config
        print(f"Running single split -> Val: {cfg.data.val_subject} | Test: {cfg.data.test_subject}")
        # Create a dummy list to satisfy the function arguments
        dummy_subjects = [cfg.data.test_subject, cfg.data.val_subject]
        run_fold(cfg, fold_idx=0, subjects=dummy_subjects)

if __name__ == "__main__":
    main()