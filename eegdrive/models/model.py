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
        loader = DataLoader(dataset, batch_size=1)
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
        def score(estimator, X, y):
            preds = estimator.predict(X)
            return np.mean(preds == y)

        for skip_channel in range(18):
            print('Remove channel', skip_channel)
            mask = np.ones(train_features.shape[1], dtype=bool)
            for i in range(100):
                mask[skip_channel * 100 + i:: 19 * 100] = False
            skip_features = train_features[:, mask]
            skip_test = test_features[:, mask]
            classifier = RidgeClassifierCV(
                alphas=(1.,), fit_intercept=False, normalize=True, scoring=score, store_cv_values=True
            )
            classifier.fit(skip_features, train_labels)
            print(classifier.cv_values_)
            predictions = classifier.predict(skip_test)
            print(predictions)
            print(test_labels)
            accuracy = np.mean(predictions == test_labels)
            print('Test accuracy:', accuracy)
            # fare cross validation su canali e poi verificare su test
            # oppure held one out
        print('All channels')
        classifier = RidgeClassifierCV(
            alphas=(1.0,), fit_intercept=False, normalize=True, scoring=score, store_cv_values=True
        )
        classifier.fit(train_features, train_labels)
        print(classifier.cv_values_)
        predictions = classifier.predict(test_features)
        print(predictions)
        print(test_labels)
        accuracy = np.mean(predictions == test_labels)
        print('Test accuracy:', accuracy)
