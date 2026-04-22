import torch
import pyro
import pyro.distributions as dist
import pyro.optim
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal, AutoMultivariateNormal
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
        self.test_acc = self.val_acc.clone()
        self._test_outputs = []
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

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
    
        self.test_acc(logits, y)
        
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)
        
        self._test_outputs.append({"logits": logits.detach(), "targets": y})

    def on_validation_epoch_end(self):
        self._shared_eval_epoch_end(self._val_outputs, prefix="val")
    
    def on_test_epoch_end(self):
        self._shared_eval_epoch_end(self._test_outputs, prefix="test")

    def on_train_end(self):
        if self._base_model_path:
            torch.save(self.model.state_dict(), self._base_model_path)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def _shared_eval_epoch_end(self, outputs, prefix):
        if not outputs:
            return
        all_logits = torch.cat([o["logits"] for o in outputs])
        all_targets = torch.cat([o["targets"] for o in outputs]).cpu().numpy()
        y_prob = torch.softmax(all_logits, dim=1).cpu().numpy()
        report, scalars = custom_classification_report(all_targets, y_prob, target_names=_TARGET_NAMES)
        print(f"\n--- {prefix.upper()} Classification Report ---\n{report}")
        self.log_dict({f"{prefix}_{k}": v for k, v in scalars.items()}, on_epoch=True, prog_bar=False)
        outputs.clear()


class BayesianClassificationTask(LightningModule):
    def __init__(self, model: torch.nn.Module, num_classes: int, lr: float, label_weights=None, pyro_checkpoint_path=None, svi_lr=1e-4, variational_family="normal"):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = torch.nn.NLLLoss(weight=label_weights)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self._val_outputs = []
        self.test_acc = self.val_acc.clone()
        self._test_outputs = []
        self._pyro_checkpoint_path = pyro_checkpoint_path
        self._best_val_acc = 0.0
        self.automatic_optimization = False

        pyro.clear_param_store()
        self.pyro_model = self._make_pyro_model()
        guide_cls = AutoMultivariateNormal if variational_family == "multivariate" else AutoNormal
        self.guide = guide_cls(self.pyro_model)
        self.svi = SVI(self.pyro_model, self.guide, pyro.optim.Adam({"lr": svi_lr}), loss=Trace_ELBO())

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
            ce_loss = self.criterion(self(x).log_softmax(dim=1), y).item()
            
            # Extract individual ELBO terms from traces to monitor posterior collapse
            guide_trace = pyro.poutine.trace(self.guide).get_trace(x, y)
            guide_trace.compute_log_prob()
            model_trace = pyro.poutine.trace(pyro.poutine.replay(self.pyro_model, trace=guide_trace)).get_trace(x, y)
            model_trace.compute_log_prob()

            log_p_obs = sum(site["log_prob"].sum().item() for name, site in model_trace.nodes.items() if site["type"] == "sample" and site["is_observed"])
            log_p_latent = sum(site["log_prob"].sum().item() for name, site in model_trace.nodes.items() if site["type"] == "sample" and not site["is_observed"])
            log_q_latent = sum(site["log_prob"].sum().item() for name, site in guide_trace.nodes.items() if site["type"] == "sample")

            kl_div = log_q_latent - log_p_latent

        self.log("elbo_loss", elbo_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", ce_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("kl_div", kl_div, on_step=False, on_epoch=True, prog_bar=True)
        self.log("log_p_obs", log_p_obs, on_step=False, on_epoch=True, prog_bar=True)

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
        loss = self.criterion(mean_probs.log(), y)
        self.val_acc(mean_probs, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self._val_outputs.append({"probs": mean_probs.detach(), "targets": y})

    def test_step(self, batch, batch_idx, n_samples=30):
        x, y = batch
        probs = []
        for _ in range(n_samples):
            guide_trace = pyro.poutine.trace(self.guide).get_trace(x, y)
            with pyro.poutine.replay(trace=guide_trace):
                logits = self(x)
            probs.append(torch.softmax(logits, dim=1))
        mean_probs = torch.stack(probs).mean(0)
        loss = self.criterion(mean_probs.log(), y)
        self.test_acc(mean_probs, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)
        self._test_outputs.append({"probs": mean_probs.detach(), "targets": y})

    def on_validation_epoch_end(self):
        self._shared_eval_epoch_end(self._val_outputs, prefix="val")

        if self._pyro_checkpoint_path:
            val_acc = float(self.trainer.callback_metrics.get("val_acc", 0.0))
            if val_acc > self._best_val_acc:
                self._best_val_acc = val_acc
                pyro.get_param_store().save(self._pyro_checkpoint_path)

    def on_test_epoch_end(self):
        self._shared_eval_epoch_end(self._test_outputs, prefix="test")

    def configure_optimizers(self):
        return []  # Pyro handles optimization via SVI

    def _shared_eval_epoch_end(self, outputs, prefix):
        if not outputs:
            return
        y_prob = torch.cat([o["probs"] for o in outputs]).cpu().numpy()
        all_targets = torch.cat([o["targets"] for o in outputs]).cpu().numpy()
        report, scalars = custom_classification_report(all_targets, y_prob, target_names=_TARGET_NAMES)
        print(f"\n--- {prefix.upper()} Classification Report ---\n{report}")
        self.log_dict({f"{prefix}_{k}": v for k, v in scalars.items()}, on_epoch=True, prog_bar=False)
        outputs.clear()
