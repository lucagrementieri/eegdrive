import numpy as np
from scipy import signal

from .eeg import EEG


class HighPass:
    def __init__(self, threshold: float, order: int = 3):
        self.threshold = threshold
        self.order = order

    def __call__(self, eeg: EEG) -> EEG:
        nyq = 0.5 * eeg.frequency
        high = self.threshold / nyq
        # noinspection PyTupleAssignmentBalance
        b, a = signal.butter(self.order, high, btype='highpass')
        for c in range(eeg.channels):
            eeg.data[c] = signal.lfilter(b, a, eeg.data[c])
        return eeg


class RemoveBeginning:
    def __init__(self, multiplier_order: int = 2):
        self.multiplier = 10 ** multiplier_order

    def __call__(self, eeg: EEG) -> EEG:
        std = np.std(eeg.data[:, eeg.length // 10:], axis=1, keepdims=True)
        anomalous = np.any(np.abs(eeg.data) > self.multiplier * std, axis=0)
        if not np.any(anomalous):
            return eeg
        cutoff = anomalous.nonzero()[0][-1] + 1
        next_preparation = np.argmax(eeg.preparation_state[cutoff:] > 0)
        next_action = np.argmax(eeg.action_state[cutoff:] > 0)
        if next_action < next_preparation:
            previous_actions = eeg.action_state[:next_preparation] > 0
            end_action = np.nonzero(previous_actions)[0][-1] + 1
            cutoff = (end_action + next_preparation) // 2
        eeg.data = eeg.data[:, cutoff:]
        eeg.preparation_state = eeg.preparation_state[cutoff:]
        eeg.action_state = eeg.action_state[cutoff:]
        return eeg


class RemoveLineNoise:
    def __init__(self, noise_frequency: float = 50.0, quality: float = 10.0):
        self.noise_frequency = noise_frequency
        self.quality = quality

    def __call__(self, eeg: EEG) -> EEG:
        nyq = 0.5 * eeg.frequency
        for f in np.arange(self.noise_frequency, nyq, self.noise_frequency):
            b, a = signal.iirnotch(f, self.quality, eeg.frequency)
            for c in range(eeg.channels):
                eeg.data[c] = signal.lfilter(b, a, eeg.data[c])
        return eeg


class Standardize:
    def __call__(self, eeg: EEG) -> EEG:
        eeg.data -= eeg.data.mean(axis=1, keepdims=True)
        eeg.data /= eeg.data.std(axis=1, keepdims=True)
        return eeg
