import torch
import torch.nn as nn
from torch.nn.functional import relu
from lightning import LightningModule
from torchmetrics.classification import MulticlassAccuracy
from deepsnooze.metrics import custom_classification_report

class DeepSleepFFNN(LightningModule):
    def __init__(self, input_size=3 * 512, num_classes=3, lr=1e-3, label_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss(weight=label_weights)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

        self.validation_step_outputs = []
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = relu(self.fc3(x))
        x = self.fc4(x)
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
        
        self.validation_step_outputs.append({"logits": logits, "targets": y})

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc(logits, y), prog_bar=True)


    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        all_logits = torch.cat([x["logits"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs]).cpu().numpy()
        y_prob = torch.softmax(all_logits, dim=1).cpu().numpy()

        print("\n" + custom_classification_report(all_targets, y_prob, target_names=['Wake', 'NREM', 'REM'], n_bins=10))
        
        # Clear memory for next epoch
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)