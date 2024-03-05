from os import PathLike

import ITrainTestEval
from Models import *
from Preprocessors import *

import pandas as pd
if __name__ == '__main__':
    pass

TRAIN_CSV_PATH: PathLike[str] = 'raw_data/fulltrain.csv'
TEST_CSV_PATH: PathLike[str] = './raw_data/balancedtest.csv'

for i in range(1, 5):
    print(f'Testing {i}-gram count vectorizer with svm')
    train_test = ITrainTestEval.ITrainTest(SvmLinearModel(), CountPreprocessor(ngram_range=(1, i)))
    print(train_test.train_test(TRAIN_CSV_PATH, TEST_CSV_PATH))
