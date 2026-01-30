from sklearn import svm

from Models.IModel import IModel


class SvmModel(IModel):
    def __init__(self):
        self.svm = svm.SVC()

    def train(self, x_train, y_train):
        self.svm.fit(x_train, y_train)

    def predict(self, x_test):
        self.svm.predict(x_test)