import json
import logging
import time
from pathlib import Path

import torch

from .ingestion import ingest_session, EpisodeDataset
from .models import FeatureExtractor1d, Model
from .utils import initialize_logger


class EEGDrive:
    @staticmethod
    def ingest(data_path: str, output_dir: str) -> None:
        initialize_logger()
        logging.info(f'Data ingestion from {data_path}')
        data_path = Path(data_path).expanduser()
        output_dir = Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        statistics = ingest_session(data_path, output_dir)
        with open(output_dir / f'{data_path.stem}_statistics.json', 'w') as f:
            json.dump(statistics, f, indent=4)

    @staticmethod
    def train(dataset_dir: str, output_dir: str, label_type: str = 'action') -> None:
        dataset_dir = Path(dataset_dir).expanduser()
        run_dir = Path(output_dir) / str(int(time.time()))
        run_dir.mkdir(parents=True)
        initialize_logger(run_dir)

        # logging.info(f'Learning rate: {lr}')

        dataset = EpisodeDataset(dataset_dir, label_type)
        feature_extractor = FeatureExtractor1d(channels=19, filters=100)
        model = Model(feature_extractor)
        torch.save(feature_extractor.state_dict(), run_dir / 'feature_extractor.pt')
        features, labels = model.represent(dataset)
        model.fit(features, labels)


"""
    @staticmethod
    def test(checkpoint: str, summary_path: str, pairs_path: str) -> Dict[str, float]:
        summary_path = Path(summary_path).expanduser()
        test_dataset = FBankDataset(summary_path)
        test_pairs = get_test_pairs(pairs_path)
        test_loader = get_test_loader(test_dataset, test_pairs)
        architecture = CustomResnet34()
        model = Model(architecture)
        model.module.load_state_dict(torch.load(checkpoint, map_location=model.device))
        scores = model.test(test_loader, test_pairs)
        return scores
"""
