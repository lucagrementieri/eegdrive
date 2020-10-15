from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class EEG:
    data: np.ndarray
    preparation_state: np.ndarray
    action_state: np.ndarray
    frequency: float = 500.0

    def slice(self, start: int, end: int) -> 'EEG':
        return EEG(
            self.data[:, start:end],
            self.preparation_state[start:end],
            self.action_state[start:end],
            self.frequency,
        )

    def split_session(self) -> Iterator['EEG']:
        start_indices = (
                np.maximum(self.preparation_state[1:] - self.preparation_state[:-1], 0) + 1
        )
        end_indices = np.append(start_indices[1:], len(self.preparation_state))
        for start, end in zip(start_indices, end_indices):
            yield self.slice(start, end)
