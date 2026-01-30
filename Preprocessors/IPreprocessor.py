import numpy as np
from enum import Enum

class IPreprocessor:

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def transform(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Processors(Enum):
    TF_IDF = 0
    TF_IDF_NGRAM = 1


def initialize_preprocessor(type: Processors):
    pass