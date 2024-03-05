from os import PathLike

import pandas as pd

from Models.IModel import IModel
from Preprocessors.IPreprocessor import IPreprocessor

from sklearn.metrics import f1_score


def split_data_csv(file_path: PathLike[str], preprocessor: IPreprocessor, train=False):
    shuffled_train_data = pd.read_csv(file_path).sample(frac=1)
    X_train = shuffled_train_data.iloc[:, 1].to_numpy()
    Y_train = shuffled_train_data.iloc[:, 0].to_numpy()
    X_train = preprocessor.fit_transform(X_train) if train else preprocessor.transform(X_train)
    print('Split shuffle done')
    return preprocessor, X_train, Y_train


class ITrainTest:

    def __init__(self, model: IModel, preprocessor: IPreprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def train_test(self, train_csv_path: PathLike[str], test_csv_path: PathLike[str]):
        self.preprocessor, X_train, Y_train = split_data_csv(train_csv_path, self.preprocessor, train=True)
        print('Done preprocessing')
        self.model.train(x_train=X_train, y_train=Y_train)
        print('Done training')
        _, X_test, Y_test = split_data_csv(test_csv_path, self.preprocessor)
        Y_pred = self.model.predict(X_test)
        return f1_score(Y_test, Y_pred, average=None)
