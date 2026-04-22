import torch
import random

class Knockout:
    def __init__(self, placeholder_value=-100.0, probability=0.3):
        """
        probability: The chance that any given channel gets knocked out.
        placeholder_value: The out-of-support value (-100.0 for log-normalized data).
        """
        self.placeholder = placeholder_value
        self.p = probability

    def __call__(self, x):
        
        M = torch.zeros_like(x, dtype=torch.bool)

        if torch.rand(1).item() < self.p:
            c = random.choice(range(x.shape[0]))
            M[c] = True  # Knock out the entire channel c

        x_aug = x.clone()
        x_aug[M] = self.placeholder
        
        return x_aug