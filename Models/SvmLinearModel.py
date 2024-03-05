import numpy as np

from Models.IModel import IModel
from sklearn.svm import LinearSVC


class SvmLinearModel(IModel):

    def __init__(self):
        self.svm_l = LinearSVC()

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        self.svm_l.fit(x_train, y_train)

    def predict(self, x_test: np.ndarray):
        return self.svm_l.predict(x_test)