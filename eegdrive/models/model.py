from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class Model:
    def __init__(self, module: nn.Module):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.module = module.to(self.device)

    def represent(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        loader = DataLoader(dataset, batch_size=1, pin_memory=True)
        self.module.eval()
        features = []
        labels = []
        for episode, label in tqdm(loader, desc='Feature extraction'):
            episode = episode.to(self.device, non_blocking=True)
            with torch.no_grad():
                episode_features = self.module(episode).cpu().numpy()
            features.append(episode_features)
            labels.append(label)
        # TODO: check size (dilation can cause problems)
        features = np.array(features)
        labels = np.array(labels)
        return features, labels

    @staticmethod
    def fit(features: np.array, labels: np.array) -> None:
        dtrain = xgb.DMatrix(features, label=labels)
        params = {
            'max_depth': 2,
            'eta': 1,
            'objective': 'multi:softmax',
            'nthread': 4,
            'eval_metric': 'merror',
        }
        results = xgb.cv(
            params,
            dtrain,
            num_boost_round=100,
            seed=42,
            nfold=5,
            metrics='merror',
            early_stopping_rounds=10,
        )
        print(results)
