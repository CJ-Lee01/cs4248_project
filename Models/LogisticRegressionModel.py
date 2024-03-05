import numpy as np

from Models.IModel import IModel
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel(IModel):

    def __init__(self):
        self.logistic_reg = LogisticRegression()

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        self.logistic_reg.fit(x_train, y_train)

    def predict(self, x_test: np.ndarray):
        return self.logistic_reg.predict(x_test)