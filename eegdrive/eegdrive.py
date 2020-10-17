import json
import time
from pathlib import Path

import torch

from .ingestion import ingest_session, EpisodeDataset
from .models import FeatureExtractor1d, Model


class EEGDrive:
    @staticmethod
    def ingest(data_path: str, output_dir: str) -> None:
        data_path = Path(data_path).expanduser()
        output_dir = Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        statistics = ingest_session(data_path, output_dir)
        with open(output_dir / f'{data_path.stem}_statistics.json', 'w') as f:
            json.dump(statistics, f, indent=4)

    @staticmethod
    def train(
            dataset_dir: str, output_dir: str, filters: int, label_type: str = 'action'
    ) -> None:
        dataset_dir = Path(dataset_dir).expanduser()
        run_dir = Path(output_dir) / str(int(time.time()))
        run_dir.mkdir(parents=True)

        dataset = EpisodeDataset(dataset_dir, label_type)
        feature_extractor = FeatureExtractor1d(channels=19, filters=filters)
        model = Model(feature_extractor)
        torch.save(feature_extractor.state_dict(), run_dir / 'feature_extractor.pt')
        features, labels = model.represent(dataset)
        model.fit(features, labels)
