"""SmallCNN: classify a (1, 32, 32) thermal crop as Person / Car / Bicycle."""
import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 8 kernels -> (B, 8, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2),  # halve H,W -> (B, 8, 16, 16)
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # -> (B, 16, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, 16, 8, 8)
            nn.Flatten(),  # -> (B, 1024)
            nn.Linear(1024, num_classes)  # one score per class
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    m = SmallCNN(num_classes=3)
    x = torch.randn(64, 1, 32, 32)  # fake batch, DataLoader shape
    for layer in m.net:  # print each layer's output shape
        x = layer(x)
        print(layer.__class__.__name__, tuple(x.shape))
