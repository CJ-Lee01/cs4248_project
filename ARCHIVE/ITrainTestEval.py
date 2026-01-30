from os import PathLike

import pandas as pd

from Models.IModel import IModel
from Preprocessors.IPreprocessor import IPreprocessor

from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler


def split_data_csv(file_path: PathLike[str], preprocessor: IPreprocessor, train=False, sep=None):
    if sep is None:
        shuffled_train_data = pd.read_csv(file_path).sample(frac=1)
    else:
        shuffled_train_data = pd.read_csv(file_path, sep=sep).sample(frac=1)
    X_train = shuffled_train_data.iloc[:, 1].to_numpy()
    Y_train = shuffled_train_data.iloc[:, 0].to_numpy()
    X_train = preprocessor.fit_transform(X_train) if train else preprocessor.transform(X_train)
    ros = RandomOverSampler()
    X_train, Y_train = ros.fit_resample(X_train, Y_train)
    return preprocessor, X_train, Y_train


class ITrainTest:

    def __init__(self, model: IModel, preprocessor: IPreprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def train_test(self, train_csv_path: PathLike[str], test_csv_path: PathLike[str], *, train_sep=None, test_sep=None):
        self.preprocessor, X_train, Y_train = split_data_csv(train_csv_path, self.preprocessor, train=True, sep=train_sep)
        self.model.train(x_train=X_train, y_train=Y_train)
        _, X_test, Y_test = split_data_csv(test_csv_path, self.preprocessor, sep=test_sep)
        print('Number of features: ', X_test.shape[1])
        Y_pred = self.model.predict(X_test)
        return [round(n, 4) for n in f1_score(Y_test, Y_pred, average=None)]


class ITrainTestSeq:
    def __init__(self, model: IModel, preprocessor: IPreprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def train_test(self, train_csv_path: PathLike[str], test_csv_path: PathLike[str], *, train_sep=None, test_sep=None):
        pass


