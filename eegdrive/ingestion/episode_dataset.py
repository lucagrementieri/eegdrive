from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class EpisodeDataset(Dataset):
    def __init__(self, dataset_dir: str):
        super().__init__()
        self.dataset_dir = Path(dataset_dir).expanduser()
        self.episodes = sorted(self.dataset_dir.glob('*.npz'))

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        path = str(self.episodes[idx])
        with np.load(path, allow_pickle=True) as npz:
            data = torch.tensor(npz['data'])
            action_label = npz['action_label']
            preparation_label = npz['preparation_label']
        return data, action_label, preparation_label
