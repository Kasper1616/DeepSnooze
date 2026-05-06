import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


class BayesianLoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha, name="lora", freeze_a=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        self.name = name
        self.std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.freeze_a = freeze_a
        if freeze_a:
            self.register_buffer("A_fixed", torch.randn(in_dim, rank) * self.std_dev)

    def forward(self, x):
        if self.freeze_a:
            A = self.A_fixed.to(x.device)
        else:
            A = pyro.sample(
                f"{self.name}_A",
                dist.Normal(0, self.std_dev).expand([self.in_dim, self.rank]).to_event(2),
            ).to(x.device)
            
        B = pyro.sample(
            f"{self.name}_B",
            dist.Normal(0, self.std_dev).expand([self.rank, self.out_dim]).to_event(2),
        ).to(x.device)
        x = self.alpha * (x @ A @ B)
        return x


class BayesianLinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha, name="lora", freeze_a=False):
        super().__init__()
        self.linear = linear
        self.lora = BayesianLoRALayer(
            linear.in_features, linear.out_features, rank, alpha, name=name, freeze_a=freeze_a
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def _disable_dropout(model):
    """Replace all dropout layers with Identity — survives model.train() calls."""
    for name, module in model.named_children():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            setattr(model, name, nn.Identity())
        else:
            _disable_dropout(module)


def apply_lora(model, rank=4, alpha=16, use_bayesian=False, freeze_a=False):
    lora_idx = 0
    for i, layer in enumerate(model.fc):
        if isinstance(layer, nn.Linear):
            if use_bayesian:
                model.fc[i] = BayesianLinearWithLoRA(layer, rank=rank, alpha=alpha, name=f"lora_{lora_idx}", freeze_a=freeze_a)
                lora_idx += 1
            else:
                model.fc[i] = LinearWithLoRA(layer, rank=rank, alpha=alpha)

    # Freeze all base params, only train LoRA params
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False
        elif freeze_a and name.endswith(".A"):
            param.requires_grad = False

    # Dropout conflicts with Bayesian posterior sampling — remove it permanently
    if use_bayesian:
        _disable_dropout(model)

    if not use_bayesian:
        lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"LoRA trainable params: {lora_params:,} / {total_params:,} ({100 * lora_params / total_params:.2f}%)"
        )

    return model


if __name__ == "__main__":
    from deepsnooze.models.cnn import SleepyCNN
    from deepsnooze.data.transforms import SpectrogramTransform

    model = SleepyCNN(num_classes=3)

    apply_lora(model, rank=1, alpha=16, use_bayesian=True)

    print(model)

    # Test the modified model with dummy input
    dummy_input = torch.randn(2, 3, 512)  # (batch_size, channels, time_steps)
    dummy_input = SpectrogramTransform()(dummy_input)  # Convert to spectrogram
    output = model(dummy_input)
    print(output.shape)  # Should be (1, 3) for num_classes=3
