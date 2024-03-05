from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

from Preprocessors.IPreprocessor import IPreprocessor


class CountPreprocessor(IPreprocessor):
    def __init__(self, ngram_range=(1, 1)):
        self.vectorizer = CountVectorizer(stop_words='english', ngram_range=ngram_range)

    def fit_transform(self, data: np.ndarray):
        return self.vectorizer.fit_transform(data)

    def transform(self, data: np.ndarray):
        return self.vectorizer.transform(data)
