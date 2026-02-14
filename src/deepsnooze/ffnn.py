import torch.nn as nn
from torch.nn.functional import relu

class SimpleFFNN(nn.Module):
    def __init__(self, input_size = 3 * 512, hidden_size = 128, num_classes = 3):
        super(SimpleFFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        return x