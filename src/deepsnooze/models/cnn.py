import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics.classification import MulticlassAccuracy
from sklearn.metrics import classification_report

class SleepyCNN(LightningModule):
    def __init__(self, num_classes=3, lr=1e-3, label_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss(weight=label_weights)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.validation_step_outputs = []

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.validation_step_outputs.append({"preds": preds, "targets": y})

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc(logits, y), prog_bar=True)

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs]).cpu().numpy()
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs]).cpu().numpy()

        print("\n" + classification_report(all_targets, all_preds, target_names=['Wake', 'NREM', 'REM'], zero_division=0))

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)