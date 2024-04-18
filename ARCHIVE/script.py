from os import PathLike

import ITrainTestEval
from Models import *
from Detectors import *

if __name__ == '__main__':
    pass

TRAIN_CSV_PATH: PathLike[str] = '../data/xtrain.txt'
TEST_CSV_PATH: PathLike[str] = '../data/balancedtest.csv'

for proc in [CountPreprocessor, DateTimePreproc,MultiPreProc, ScalerPreProc]:
    print(f'Testing {proc().__name__()} with logistic regression')
    train_test = ITrainTestEval.ITrainTest(LogisticRegressionModel(), proc())
    print('Macro f1:', "%.5f" % train_test.train_test(TRAIN_CSV_PATH, TEST_CSV_PATH, train_sep='\t'))

