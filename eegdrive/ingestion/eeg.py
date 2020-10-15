from dataclasses import dataclass

import numpy as np


@dataclass
class EEG:
    data: np.ndarray
    preparation_state: np.ndarray
    action_state: np.ndarray
    frequency: float = 500.0
