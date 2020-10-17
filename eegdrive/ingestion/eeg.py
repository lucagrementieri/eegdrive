from dataclasses import dataclass
from pathlib import Path
from typing import Union, Iterator

import h5py
import numpy as np


@dataclass
class EEG:
    data: np.ndarray
    preparation_state: np.ndarray
    action_state: np.ndarray
    frequency: float = 500.0

    @staticmethod
    def from_hdf5(path: Union[Path, str], frequency: float = 500.0) -> 'EEG':
        with h5py.File(path, 'r') as h5f:
            return EEG(
                np.array(h5f['eeg_data']).T,
                np.array(h5f['preparation_state'], dtype=np.int64),
                np.array(h5f['action_state'], dtype=np.int64),
                frequency,
            )

    def __len__(self):
        return self.data.shape[1]

    @property
    def channels(self):
        return self.data.shape[0]

    @property
    def length(self):
        return len(self)

    def slice(self, start: int, end: int) -> 'EEG':
        return EEG(
            self.data[:, start:end],
            self.preparation_state[start:end],
            self.action_state[start:end],
            self.frequency,
        )

    def split_session(self) -> Iterator['EEG']:
        diff = self.preparation_state[1:] - self.preparation_state[:-1]
        start_indices = (diff * (diff > 0)).nonzero()[0] + 1
        end_indices = np.append(start_indices[1:], len(self))
        for start, end in zip(start_indices, end_indices):
            assert self.preparation_state[start] > 0
            assert start == 0 or self.preparation_state[start - 1] == 0
            assert self.preparation_state[end - 1] == 0
            assert end == len(self) or self.preparation_state[end + 1] > 0
            yield self.slice(start, end)
