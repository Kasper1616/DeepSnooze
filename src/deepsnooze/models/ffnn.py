import torch
import torch.nn as nn
from lightning import LightningModule


class DeepSleepFFNN(LightningModule):
    def __init__(self, input_size=3 * 512, num_classes=3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.fc(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
