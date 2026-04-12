import torch.nn as nn


def _conv_block(in_ch, out_ch, dropout=0.3):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Dropout2d(dropout),
        nn.MaxPool2d(2, 2),
    )


class SimpleCNN(nn.Module):
    """2 conv blocks — minimal baseline."""

    def __init__(self, num_classes=3, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.features = nn.Sequential(
            _conv_block(3, 16),
            _conv_block(16, 32),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(self.features(x))


class SleepyCNN(nn.Module):
    """3 conv blocks — medium baseline."""

    def __init__(self, num_classes=3, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(self.features(x))


class DeepCNN(nn.Module):
    """4 conv blocks — advanced baseline."""

    def __init__(self, num_classes=3, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.fc(self.features(x))
