import torch
import pyro
import pyro.distributions as dist
import pyro.optim
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from lightning import LightningModule
from torchmetrics.classification import MulticlassAccuracy

from deepsnooze.evaluation.metrics import custom_classification_report
from deepsnooze.losses import FocalLoss

_TARGET_NAMES = ["Wake", "NREM", "REM"]


class StandardClassificationTask(LightningModule):
    def __init__(self, model: torch.nn.Module, num_classes: int, lr: float, label_weights=None,
                 base_model_path=None, loss="ce", label_smoothing=0.0, focal_gamma=2.0):
        super().__init__()
        self.model = model
        self.lr = lr
        if loss == "focal":
            self.criterion = FocalLoss(gamma=focal_gamma, weight=label_weights)
        else:
            smoothing = label_smoothing if loss == "label_smoothing" else 0.0
            self.criterion = torch.nn.CrossEntropyLoss(weight=label_weights, label_smoothing=smoothing)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self._val_outputs = []
        self._base_model_path = base_model_path

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self._val_outputs.append({"logits": logits.detach(), "targets": y})

    def on_validation_epoch_end(self):
        if not self._val_outputs:
            return
        all_logits = torch.cat([o["logits"] for o in self._val_outputs])
        all_targets = torch.cat([o["targets"] for o in self._val_outputs]).cpu().numpy()
        y_prob = torch.softmax(all_logits, dim=1).cpu().numpy()
        report, scalars = custom_classification_report(all_targets, y_prob, target_names=_TARGET_NAMES)
        print("\n" + report)
        self.log_dict(scalars, on_epoch=True, prog_bar=False)
        self._val_outputs.clear()

    def on_train_end(self):
        if self._base_model_path:
            torch.save(self.model.state_dict(), self._base_model_path)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


class BayesianClassificationTask(LightningModule):
    def __init__(self, model: torch.nn.Module, num_classes: int, lr: float, label_weights=None, pyro_checkpoint_path=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = torch.nn.CrossEntropyLoss(weight=label_weights)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self._val_outputs = []
        self._pyro_checkpoint_path = pyro_checkpoint_path
        self._best_val_acc = 0.0
        self.automatic_optimization = False

        pyro.clear_param_store()
        self.pyro_model = self._make_pyro_model()
        self.guide = AutoNormal(self.pyro_model)
        self.svi = SVI(self.pyro_model, self.guide, pyro.optim.Adam({"lr": lr * 0.1}), loss=Trace_ELBO())

    def _make_pyro_model(self):
        net = self.model

        def model(x, y=None):
            logits = net(x)
            with pyro.plate("data", x.shape[0]):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
            return logits

        return model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        elbo_loss = float(self.svi.step(x, y))
        with torch.no_grad():
            ce_loss = self.criterion(self(x), y).item()
        self.log("elbo_loss", elbo_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", ce_loss, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx, n_samples=30):
        x, y = batch
        # Average softmax probabilities over multiple posterior samples
        probs = []
        for _ in range(n_samples):
            guide_trace = pyro.poutine.trace(self.guide).get_trace(x, y)
            with pyro.poutine.replay(trace=guide_trace):
                logits = self(x)
            probs.append(torch.softmax(logits, dim=1))
        mean_probs = torch.stack(probs).mean(0)
        # Reconstruct logits from mean probs for loss/acc compatibility
        mean_logits = mean_probs.log()
        loss = self.criterion(mean_logits, y)
        self.val_acc(mean_logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self._val_outputs.append({"probs": mean_probs.detach(), "targets": y})

    def on_validation_epoch_end(self):
        if not self._val_outputs:
            return
        y_prob = torch.cat([o["probs"] for o in self._val_outputs]).cpu().numpy()
        all_targets = torch.cat([o["targets"] for o in self._val_outputs]).cpu().numpy()
        report, scalars = custom_classification_report(all_targets, y_prob, target_names=_TARGET_NAMES)
        print("\n" + report)
        self.log_dict(scalars, on_epoch=True, prog_bar=False)
        self._val_outputs.clear()

        if self._pyro_checkpoint_path:
            val_acc = float(self.trainer.callback_metrics.get("val_acc", 0.0))
            if val_acc > self._best_val_acc:
                self._best_val_acc = val_acc
                pyro.get_param_store().save(self._pyro_checkpoint_path)

    def configure_optimizers(self):
        return []  # Pyro handles optimization via SVI
