import torch
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
import pyro.optim  # type: ignore[reportAttributeAccessIssue]
from torchmetrics.classification import MulticlassAccuracy

from sklearn.utils.class_weight import compute_class_weight

from deepsnooze.data_module import SleepDataModule, SleepyRatDataset
from deepsnooze.models.ffnn import DeepSleepFFNN
from deepsnooze.models.cnn import SleepyCNN
from deepsnooze.transforms import SpectrogramTransform
from deepsnooze.models.lora import apply_lora
from deepsnooze.metrics import custom_classification_report


def make_pyro_model(net):
    def model(x, y=None):
        pyro.module("net", net)
        logits = net(x)
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits

    return model


def train_vi_bayesian(
    model_type="cnn", max_epochs=20, batch_size=32, lr=1e-3, rank=1, alpha=10, device="cuda"
):
    datamodule = SleepDataModule(
        processed_path="data/processed",
        batch_size=batch_size,
        val_subject="A1",
        test_subject="D6",
        transform=SpectrogramTransform(),
    )
    datamodule.setup(stage="fit")

    full_ds: SleepyRatDataset = datamodule.train_ds.dataset  # type: ignore[assignment]
    all_labels = np.array(full_ds.labels)
    train_labels = all_labels[datamodule.train_ds.indices]

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(train_labels), y=train_labels
    )
    label_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Calculated Class Weights: {label_weights}")

    if model_type == "ffnn":
        base_model = DeepSleepFFNN(lr=lr, label_weights=label_weights)
    else:
        base_model = SleepyCNN(lr=lr, label_weights=label_weights)

    checkpoint = torch.load(
        f"models/{base_model.__class__.__name__}.ckpt", weights_only=False
    )
    base_model.load_state_dict(checkpoint["state_dict"], strict=False)

    apply_lora(base_model, rank=rank, alpha=alpha, use_bayesian=True)
    base_model.to(device)

    pyro.clear_param_store()

    pyro_model = make_pyro_model(base_model)
    guide = AutoNormal(pyro_model)
    optimizer = pyro.optim.Adam({"lr": lr})
    svi = SVI(pyro_model, guide, optimizer, loss=Trace_ELBO())

    val_acc_metric = MulticlassAccuracy(num_classes=3).to(device)
    best_val_acc = 0.0

    for epoch in range(max_epochs):
        # Training
        base_model.train()
        epoch_loss = 0.0
        num_batches = 0
        for x, y in datamodule.train_dataloader():
            x, y = x.to(device), y.to(device)
            loss = float(svi.step(x, y))
            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        base_model.eval()
        val_acc_metric.reset()
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for x, y in datamodule.val_dataloader():
                x, y = x.to(device), y.to(device)
                guide_trace = pyro.poutine.trace(guide).get_trace(x, y)
                with pyro.poutine.replay(trace=guide_trace):
                    logits = base_model(x)
                val_acc_metric.update(logits, y)
                all_labels.append(y.cpu().numpy())
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

        val_acc = val_acc_metric.compute().item()
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)

        report = custom_classification_report(
            all_labels, all_probs, target_names=["Wake", "NREM", "REM"]
        )
        print(f"Epoch {epoch + 1}/{max_epochs} | ELBO loss: {avg_loss:.4f} | val_acc: {val_acc:.4f}")
        print(report)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Log ELBO loss and val_acc to csv
        with open(f"logs/{base_model.__class__.__name__}_bayesian_lora_k{rank}.csv", "a") as f:
            if epoch == 0:
                f.write("epoch,elbo_loss,val_acc\n")
            f.write(f"{epoch + 1},{avg_loss:.4f},{val_acc:.4f}\n")

    pyro.get_param_store().save(
        f"models/{base_model.__class__.__name__}_bayesian_lora_k{rank}.pt"
    )
    print(f"Training complete. Best val_acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Bayesian LoRA for sleep stage classification."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "ffnn"],
        help="Base model type.",
    )
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=10)

    args = parser.parse_args()
    train_vi_bayesian(
        model_type=args.model,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        rank=args.rank,
        alpha=args.alpha,
    )
