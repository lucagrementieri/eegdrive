import itertools
from typing import Tuple, List

import torch
from torch import nn


class RandomConv1d(nn.Module):
    def __init__(
            self,
            channels: int,
            filters: int,
            sizes: Tuple[int, ...] = (7, 9, 11),
            max_dilation_exponent: int = 8,
    ):
        super().__init__()
        self.convolutions = nn.ModuleList()
        for k, d_exp, pad in itertools.product(
                sizes, range(max_dilation_exponent + 1), (True, False)
        ):
            padding = (k - 1) // 2 if pad else 0
            self.convolutions.append(
                nn.Conv1d(
                    channels,
                    filters,
                    k,
                    padding=padding,
                    dilation=2 ** d_exp,
                    groups=channels,
                )
            )
        self.random_weights()

    def random_weights(self) -> None:
        for conv in self.convolutions:
            nn.init.normal_(conv.weight)
            conv.weight -= conv.weight.mean(dim=-1, keepdim=False)
            nn.init.uniform_(conv.bias, -1, 1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [conv(x) for conv in self.convolutions]
