import numpy as np


class IModel:

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        raise NotImplementedError

    def predict(self, x_test: np.ndarray):
        raise NotImplementedError
