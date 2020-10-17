from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class EpisodeDataset(Dataset):
    def __init__(self, dataset_dir: str, label_type: str):
        super().__init__()
        if label_type not in ('action', 'preparation'):
            raise ValueError('Invalid label type')
        self.dataset_dir = Path(dataset_dir).expanduser()
        self.label_type = label_type
        self.episodes = sorted(self.dataset_dir.glob('*.npz'))

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = str(self.episodes[idx])
        with np.load(path, allow_pickle=True) as npz:
            data = torch.tensor(npz['data'])
            label = npz[f'{self.label_type}_label']
        return data, label
