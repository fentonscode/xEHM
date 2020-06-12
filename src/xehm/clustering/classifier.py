import numpy as np
from typing import Tuple

__all__ = ["Classifier"]


class Classifier:
    def __init__(self):
        if type(self) is Classifier:
            raise Exception("Classifier is an abstract base, inherit and define your own")
        self.n_groups = 0
        self.centres = []

    def assign(self, data: np.ndarray, parameters: Tuple):
        raise NotImplementedError("Custom classifiers must implement assign")

    def classify(self, data: np.ndarray) -> int:
        raise NotImplementedError("Custom classifiers must implement assign")

    def __len__(self):
        return self.n_groups

    def __getitem__(self, item):
        return self.centres[item]
