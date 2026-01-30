from os import PathLike
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model

import ITrainTestEval
from Detectors import DateTimePreproc, MultiPreProc, ScalerPreProc
from Models import *
from Preprocessors import CountPreprocessor, TfIdfPreprocessor

if __name__ == "__main__":
    pass

report = open('REPORT', 'a')
report.write(f'\nPreprocessors against logistic regression\n')

TRAIN_CSV_PATH: PathLike[str] = '../data/xtrain.txt'
TEST_CSV_PATH: PathLike[str] = '../data/balancedtest.csv'


def format_arr(arr: List[Tuple[float, str]]):
    return '\t'.join([f'{word}: {float(n):.4f}' for n, word in arr])

for proc in [TfIdfPreprocessor]:
    """print(f'Testing {proc().__name__()} with logistic regression')
    report.write(f'{proc().__name__()}\n')"""
    processor = proc()
    model = LogisticRegressionModel()
    train_test = ITrainTestEval.ITrainTest(model, processor)
    loss = train_test.train_test(TRAIN_CSV_PATH, TEST_CSV_PATH, train_sep='\t')
    """print(f"f1: {str(loss)}")
    report.write(f'f1 score: {str(loss)}\n')"""
    df = pd.read_csv(TEST_CSV_PATH)
    text = df.iloc[:, 1]
    df[f'{proc().__name__()}'] = model.predict(processor.transform(text.to_numpy()))
    df.to_csv(TEST_CSV_PATH, index=False)
    df_seri = pd.read_csv('../data/NewsRealCOVID-19.csv')
    seri = (df_seri['title'] + df_seri['content'].fillna('').apply(lambda x: ' ' + x)).apply(lambda x: x.lower().replace('vaccine', ''))
    test_text = list(seri[~seri.isnull()])
    tokenized_test_text = processor.vectorizer.build_analyzer()(test_text[0])
    vectorized_text = processor.transform(test_text)
    predictions = (model.predict(vectorized_text))
    confidences = model.logistic_reg.predict_proba(vectorized_text)
    print(vectorized_text.shape)
    print(predictions)
    print(confidences)
    counts = [0,0,0,0]
    """
    c1, c2, c3, c4 = model.logistic_reg.coef_
    vocab = processor.vectorizer.vocabulary_
    vocabs = [''] * len(vocab)
    for key, val in vocab.items():
        vocabs[val] = key

    vals = c4 * vectorized_text.toarray()[0]
    result = []
    for i in set(tokenized_test_text):
        i = i.lower()
        if i not in vocab:
            result.append((i + ' (UNK)', round(vals[0], 4)))
            continue
        result.append((i, round(vals[vocab[i]], 4)))
    print(4, sorted(result, key=lambda x: x[1], reverse=True))

    vals = c3 * vectorized_text.toarray()[0]
    result = []
    for i in set(tokenized_test_text):
        i = i.lower()
        if i not in vocab:
            result.append((i + ' (UNK)', round(vals[0], 4)))
            continue
        result.append((i, round(vals[vocab[i]], 4)))
    print(3, sorted(result, key=lambda x: x[1], reverse=True))
    break"""
    for i in predictions:
        counts[i - 1] += 1


    import matplotlib.pyplot as pyplot

    for i in range(4):
        fig, ax = plt.subplots()
        ax.set_xlabel("Confidence (Probability)")
        ax.set_ylabel("Number of articles")
        ax.set_title(f"Confidence for class {i + 1}")
        pyplot.hist(confidences[:, i], bins=100)
        pyplot.show()

    print(counts)
    df_n = {'text': test_text, 'label': predictions, 'confidence_1': confidences[:, 0], 'confidence_2': confidences[:, 1], 'confidence_3': confidences[:, 2], 'confidence_4': confidences[:, 3]}
    df_n = pd.DataFrame(df_n)
    df_n.to_csv('../data/ReliableCovid_NOVAC.csv')

    """c1, c2, c3, c4 = model.logistic_reg.coef_
    vocab = processor.vectorizer.vocabulary_
    vocabs = [''] * len(vocab)
    for key, val in vocab.items():
        vocabs[val] = key

    c_z1 = sorted(zip(c1, vocabs))
    c_z2 = sorted(zip(c2, vocabs))
    c_z3 = sorted(zip(c3, vocabs))
    c_z4 = sorted(zip(c4, vocabs))
    tmep = [c_z1, c_z2, c_z3, c_z4]
    for i in range(4):
        report.write(f'For class {i + 1}\n')
        report.write('Bottom 20:\n')
        report.write(format_arr(tmep[i][:20]) + '\n\n')
        report.write('Top 20:\n')
        report.write(format_arr(tmep[i][-20:]) + '\n\n')"""



report.write('\nSVM\n')

