from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
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
                episode_features = self.module(episode).squeeze_().cpu().numpy()
            features.append(episode_features)
            labels.append(label)
        features = np.array(features)
        labels = np.array(labels)
        return features, labels

    @staticmethod
    def fit(features: np.array, labels: np.array) -> None:
        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.09, random_state=42
        )
        classifier = RidgeClassifierCV(
            alphas=(0.01, 0.025, 0.05, 0.075, 0.1), fit_intercept=False, normalize=True, cv=6
        )
        classifier.fit(train_features, train_labels)
        print(classifier.best_score_)
        print(classifier.alpha_)
        print(classifier.predict(test_features))
        print(test_labels)
