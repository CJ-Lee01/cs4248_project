import numpy as np

from Preprocessors.IPreprocessor import IPreprocessor
from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfPreprocessor(IPreprocessor):
    def __init__(self, ngram_range=(1, 1)):
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)

    def fit_transform(self, data: np.ndarray):
        return self.vectorizer.fit_transform(data)

    def transform(self, data: np.ndarray):
        return self.vectorizer.transform(data)

    def __name__(self):
        return 'tf-idf baseline'
