import numpy as np

from Models.IModel import IModel
from sklearn import svm
from sklearn.metrics import f1_score


class SvmMultiParallelModel(IModel):

    def __init__(self, num_cat=4):
        self.svm_arr = []
        self.num_arr = []

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        # Use sth like a random forest: multiple svms in parallel.
        # prioritize SVMs with better performance
        y_unique = np.unique(y_train)
        for i in y_unique:
            y_tr_temp = (y_train == i).astype('float')
            i_svc = svm.SVC()
            i_svc.fit(x_train, y_tr_temp)
            i_svc_score = f1_score(i_svc.predict(x_train), y_tr_temp)
            self.svm_arr.append((i_svc_score, i, i_svc))
            print('Done SVM for', i)
        self.svm_arr = sorted(self.svm_arr, key=lambda s: s[0:2], reverse=True)
        self.num_arr = np.array([i[1] for i in self.svm_arr])

    def predict(self, x_test: np.ndarray):
        assert len(self.svm_arr) > 0, 'Model has not been trained yet!'
        predictions = np.array([svc.predict(x_test) for svc in self.svm_arr]).T
        predictions[:, -1] = 1 # Default prediction
        predictions = self.num_arr[np.argmax(predictions)]
        return predictions
