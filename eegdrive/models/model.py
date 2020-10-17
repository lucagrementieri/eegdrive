from typing import Union, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
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
                episode_features = self.module(episode).squeeze_().cpu().numpy()
            features.append(episode_features)
            labels.append(label)
        features = np.array(features)
        labels = np.array(labels)
        return features, labels

    def channel_selection(self, features: np.ndarray, labels: np.ndarray):
        features = features.reshape(features.shape[0], -1)
        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.09, random_state=42,
        )

        def accuracy(estimator: RidgeClassifier, x: np.ndarray, y: np.ndarray):
            return np.mean(estimator.predict(x) == y)

        best_accuracy = 0
        iteration_accuracies = []
        channel_mask = np.ones(self.module.channels, dtype=bool)
        selected_features = train_features
        excluded_channels = []
        while True:
            for channel in trange(channel_mask.sum(), desc='Channel selection'):
                pruned_features = self._remove_channel_features(
                    selected_features, channel
                )
                cv_accuracies = cross_val_score(
                    self.classifier,
                    pruned_features,
                    train_labels,
                    scoring=accuracy,
                    cv=6,
                )
                iteration_accuracies.append(np.array(cv_accuracies).mean())
            iteration_accuracies = np.array(iteration_accuracies)
            print(iteration_accuracies)
            best_iteration_channel = np.argmax(iteration_accuracies)
            best_iteration_accuracy = iteration_accuracies[best_iteration_channel]
            if best_iteration_accuracy < best_accuracy:
                break
            best_accuracy = best_iteration_accuracy
            iteration_accuracies = []
            excluded_channels.append(best_iteration_channel)
            corrected_channel_idx = np.argmax(
                channel_mask.cumsum() == best_iteration_channel
            )
            assert channel_mask[corrected_channel_idx]
            channel_mask[corrected_channel_idx] = False
            selected_features = self._remove_channel_features(
                selected_features, best_iteration_channel
            )
        print('Original algorithm', excluded_channels)
        excluded_channels = (~channel_mask).nonzero()[0]
        print('Corrected excluded')
        selected_train_features = self._remove_channel_features(
            train_features, excluded_channels
        )
        assert np.all(selected_train_features == selected_features)
        selected_test_features = self._remove_channel_features(
            test_features, excluded_channels
        )
        self.classifier.fit(selected_train_features, train_labels)
        predictions = self.classifier.predict(selected_test_features)
        print(predictions)
        print(test_labels)
        accuracy = np.mean(predictions == test_labels)
        print('Test accuracy:', accuracy)


"""
    @staticmethod
    def fit(features: np.ndarray, labels: np.ndarray) -> None:
        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.09, random_state=42
        )

        def score(estimator, X, y):
            preds = estimator.predict(X)
            return np.mean(preds == y)

        classifier = RidgeClassifier(fit_intercept=False, normalize=True)
        for skip_channel in range(18):
            print('Remove channel', skip_channel)
            mask = np.ones(train_features.shape[1], dtype=bool)
            for i in range(100):
                mask[skip_channel * 100 + i:: 19 * 100] = False
            skip_features = train_features[:, mask]
            skip_test = test_features[:, mask]
            cv_scores = cross_val_score(
                classifier, skip_features, train_labels, scoring=score, cv=6
            )
            print(cv_scores)
            classifier.fit(skip_features, train_labels)
            predictions = classifier.predict(skip_test)
            print(predictions)
            print(test_labels)
            accuracy = np.mean(predictions == test_labels)
            print('Test accuracy:', accuracy)
        print('All channels')
        classifier = RidgeClassifier(fit_intercept=False, normalize=True)
        classifier.fit(train_features, train_labels)
        predictions = classifier.predict(test_features)
        print(predictions)
        print(test_labels)
        accuracy = np.mean(predictions == test_labels)
        print('Test accuracy:', accuracy)
"""
