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
        sizes: Tuple[int, ...] = (7, 9, 11),
        max_dilation_exponent: int = 7,
    ):
        super().__init__()
        self.channels = channels
        self.filters = filters
        self.sizes = sizes
        self.max_dilation_exponent = max_dilation_exponent
        self.n_layers = len(sizes) * (max_dilation_exponent + 1) * 2

        self.random_conv = RandomConv1d(channels, filters, sizes, max_dilation_exponent)
        self.max_pool = GlobalMaxPool(dim=-1)
        self.proportion_pool = PositiveProportion(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        responses = self.random_conv(x)
        max_features = torch.cat([
            self.max_pool(r).reshape(x.shape[0], self.channels, self.filters)
            for r in responses
        ], dim=-1)
        proportion_features = torch.cat([
            self.proportion_pool(r).reshape(x.shape[0], self.channels, self.filters)
            for r in responses
        ], dim=-1)
        features = torch.cat((max_features, proportion_features), dim=-1)
        return features
