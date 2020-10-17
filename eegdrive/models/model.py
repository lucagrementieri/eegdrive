from typing import Union, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange


class Model:
    def __init__(self, module: nn.Module):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.module = module.to(self.device)
        self.classifier = RidgeClassifier(fit_intercept=False, normalize=True)

    def _remove_channel_features(
            self, features: np.ndarray, channel: Union[int, Sequence[int]]
    ) -> np.ndarray:
        selected_features = features.reshape(
            (features.shape[0], -1, 2 * self.module.n_layers * self.module.filters)
        )
        selected_features = np.delete(selected_features, channel, axis=1)
        selected_features = selected_features.reshape(features.shape[0], -1)
        return selected_features

    def represent(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        loader = DataLoader(dataset, batch_size=1)
        self.module.eval()
        features = []
        labels = []
        for episode, label in tqdm(loader, desc='Feature extraction'):
            episode = episode.to(self.device, non_blocking=True)
            with torch.no_grad():
                episode_features = self.module(episode).flatten().cpu().numpy()
            features.append(episode_features)
            labels.append(label)
        features = np.array(features)
        labels = np.array(labels)
        return features, labels

    def channel_selection(
            self, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        def accuracy(estimator: RidgeClassifier, x: np.ndarray, y: np.ndarray) -> float:
            return float(np.mean(estimator.predict(x) == y))

        best_accuracy = 0
        iteration_accuracies = []
        channel_mask = np.ones(self.module.channels, dtype=bool)
        selected_features = features
        while True:
            for channel in trange(channel_mask.sum(), desc='Channel selection'):
                pruned_features = self._remove_channel_features(
                    selected_features, channel
                )
                cv_accuracies = cross_val_score(
                    self.classifier, pruned_features, labels, scoring=accuracy, cv=5,
                )
                iteration_accuracies.append(np.array(cv_accuracies).mean())
            iteration_accuracies = np.array(iteration_accuracies)
            print(iteration_accuracies)
            best_iteration_channel = np.argmax(iteration_accuracies)
            best_iteration_accuracy = iteration_accuracies[best_iteration_channel]
            if best_iteration_accuracy - 0.01 < best_accuracy:
                break
            best_accuracy = best_iteration_accuracy
            iteration_accuracies = []
            corrected_channel_idx = np.argmax(
                channel_mask.cumsum() - 1 == best_iteration_channel
            )
            assert channel_mask[corrected_channel_idx]
            channel_mask[corrected_channel_idx] = False
            selected_features = self._remove_channel_features(
                selected_features, best_iteration_channel
            )
        excluded_channels = (~channel_mask).nonzero()[0]
        return excluded_channels, best_accuracy

    def fit(
            self,
            features: np.ndarray,
            labels: np.ndarray,
            excluded_channels: Optional[np.ndarray] = None,
    ) -> None:
        selected_features = self._remove_channel_features(features, excluded_channels)
        self.classifier.fit(selected_features, labels)

    def predict(
            self, features: np.ndarray, excluded_channels: Optional[np.ndarray] = None
    ) -> np.ndarray:
        selected_features = self._remove_channel_features(features, excluded_channels)
        return self.classifier.predict(selected_features)

    def eval(
            self,
            features: np.ndarray,
            labels: np.ndarray,
            excluded_channels: Optional[np.ndarray] = None,
    ) -> float:
        predictions = self.predict(features, excluded_channels)
        accuracy = float(np.mean(predictions == labels))
        return accuracy
