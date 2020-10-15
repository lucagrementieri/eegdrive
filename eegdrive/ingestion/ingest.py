from pathlib import Path

from .eeg import EEG
from .transforms import HighPass, RemoveBeginning, RemoveLineNoise, Standardize
import numpy as np
from tqdm import tqdm


def ingest_session(data_path: Path, output_dir: Path):
    eeg = EEG.from_hdf5(data_path)
    transforms = [HighPass(1.0), RemoveBeginning(), RemoveLineNoise()]
    for transform in transforms:
        eeg = transform(eeg)
    standardize = Standardize()
    for i, episode in enumerate(tqdm(eeg.split_session(), desc='Ingestion')):
        episode = standardize(episode)
        action_label = np.max(np.unique(episode.action_state))
        preparation_label = np.max(np.unique(episode.preparation_state))
        activity = episode.action_state + episode.preparation_state
        min_length = activity.nonzero()[0][-1] + 1
        np.savez(
            output_dir / f'{data_path.stem}_{i:04d}.npz',
            data=episode.data,
            action_label=action_label,
            preparation_label=preparation_label,
            min_length=min_length,
        )
