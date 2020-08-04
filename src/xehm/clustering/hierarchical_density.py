import hdbscan as hd
import numpy as np
from typing import Tuple
from .classifier import Classifier


class HDCluster(Classifier):
    def __init__(self):
        super().__init__()

    def assign(self, data: np.ndarray, parameters: Tuple):
        h_object: hd.HDBSCAN = hd.HDBSCAN().fit(X=data)
        self.n_groups = max(h_object.labels_)

    def classify(self, data: np.ndarray) -> int:
        pass
