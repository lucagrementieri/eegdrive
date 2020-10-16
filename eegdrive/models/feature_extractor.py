from typing import Tuple

import torch
import torch.nn as nn

from .pooling import GlobalMaxPool, PositiveProportion
from .random_conv import RandomConv1d


class FeatureExtractor1d(nn.Module):
    def __init__(
            self,
            channels: int,
            filters: int,
            sizes: Tuple[int, ...],
            max_dilation_exponent: int = 8,
    ):
        super().__init__()
        self.random_conv = RandomConv1d(channels, filters, sizes, max_dilation_exponent)
        self.max_pool = GlobalMaxPool(dim=-1)
        self.positive_pool = PositiveProportion(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        responses = self.random_conv(x)
        features = [
            pool(r) for r in responses for pool in (self.max_pool, self.positive_pool)
        ]
        return torch.cat(features, dim=-1)
