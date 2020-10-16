from pathlib import Path
from typing import Union, Dict, List

import numpy as np

from .eeg import EEG
from .transforms import HighPass, RemoveBeginning, RemoveLineNoise, Standardize


def ingest_session(
        data_path: Path, output_dir: Path
) -> Dict[str, Union[int, List[int]]]:
    eeg = EEG.from_hdf5(data_path)
    transforms = [HighPass(1.0), RemoveBeginning(), RemoveLineNoise(), Standardize()]
    for transform in transforms:
        eeg = transform(eeg)
    statistics = {
        'dataset_size': 0,
        'action_count': [0] * 5,
        'preparation_count': [0] * 5,
        'essential_length_q': None,
        'length_q': None,
    }
    lengths = []
    essential_lengths = []
    for i, episode in enumerate(eeg.split_session()):
        action_label = np.max(np.unique(episode.action_state))
        preparation_label = np.max(np.unique(episode.preparation_state))
        activity = episode.action_state + episode.preparation_state
        essential_length = activity.nonzero()[0][-1] + 1
        np.savez(
            output_dir / f'{data_path.stem}_{i:03d}.npz',
            data=episode.data,
            action_label=action_label,
            preparation_label=preparation_label,
            essential_length=essential_length,
        )
        lengths.append(episode.data.shape[1])
        essential_lengths.append(essential_length)
        statistics['dataset_size'] += 1
        statistics['action_count'][action_label] += 1
        statistics['preparation_count'][max(0, preparation_label - 4)] += 1
    statistics['essential_length_q'] = np.quantile(
        essential_lengths, (0, 0.25, 0.5, 0.75, 1)
    )
    statistics['essential_length_q'] = (
        statistics['essential_length_q'].astype(np.int64).tolist()
    )
    statistics['length_q'] = np.quantile(lengths, (0, 0.25, 0.5, 0.75, 1))
    statistics['length_q'] = statistics['length_q'].astype(np.int64).tolist()
    return statistics
