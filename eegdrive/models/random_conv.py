import itertools
from typing import Tuple, List

import torch
from torch import nn
import torch.nn.functional as F


class RandomConv1d(nn.Module):
    def __init__(
            self,
            channels: int,
            filters: int,
            sizes: Tuple[int, ...] = (7, 9, 11),
            max_dilation_exponent: int = 7,
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
                    filters * channels,
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
            conv.weight.data -= conv.weight.mean(dim=-1, keepdim=True)
            nn.init.uniform_(conv.bias, -1, 1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for conv in self.convolutions:
            d = conv.weight.shape[-1] * conv.dilation[-1] - 1 - x.shape[-1]
            if d > 0:
                padding_left = d // 2
                padding_right = d - padding_left
                outputs.append(conv(F.pad(x, [padding_left, padding_right])))
            else:
                outputs.append(conv(x))
        return outputs
