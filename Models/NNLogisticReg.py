import numpy as np

from Models.IModel import IModel
from sklearn.neural_network import MLPClassifier

class NNLogisticRegressionModel(IModel):

    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(10, 10))

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(x_train, y_train)

    def predict(self, x_test: np.ndarray):
        self.model.predict(x_test)


