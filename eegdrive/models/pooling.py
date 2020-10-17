import torch
import torch.nn as nn


class PositiveProportion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.tensor(x > 0).to(torch.float32), dim=self.dim)


class GlobalMaxPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(x, dim=self.dim).values
