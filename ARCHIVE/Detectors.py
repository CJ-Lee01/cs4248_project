import numpy as np
import scipy.sparse
import sklearn.preprocessing
from nltk import wordpunct_tokenize
import re

from Preprocessors import IPreprocessor, TfIdfPreprocessor, CountPreprocessor

date_time_words = {
    "year", "month", "day", "hour", "minute", "second",
    "weekday", "week", "quarter", "timezone", "timestamp",
    "duration", "epoch", "format", "weekend",
    "period", "zone", "interval"
}

months_set = {s.lower() for s in [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]}

months_set_2 = {s.lower() for s in [
    "Jan", "Feb", "Mar", "Apr",
    "May", "Jun", "Jul", "Aug",
    "Sep", "Oct", "Nov", "Dec"
]}

def is_date(s: str):
    """
    Checks if the dates are in the following formats: {
        DD/MM, DD-MM, MM/DD, MM-DD,
        Month_word DD, DD Month_word,
        DD-MM-YYYY, DD/MM/YYYY, MM-DD-YYYY, MM/DD/YYYY
        Month_word DD YYYY, DD Month_word YYYY
        Month_word YYYY
    }
    :param s: the date string
    :return: True if it is a date string, false otherwise
    """
    # DD/MM or DD-MM or MM-DD or MM-DD
    if re.fullmatch('\d?\d[/-]\d?\d', s):
        # no complicated logic here
        part_1, part_2 = re.findall('\d+', s)
        part_1, part_2 = int(part_1), int(part_2)
        return (part_1 < 32 and part_2 < 13) or (part_1 < 13 and part_2 < 32)

    # DD Month YYYY
    if re.fullmatch('\d?\d[\s/-][a-zA-Z]+[\s/-]\d{1,4}]', s):
        part_1, part_2, part_3 = re.findall('\w+', s)
        part_1, part_2, part_3 = int(part_1), part_2.lower(), int(part_3)
        if part_2 not in months_set and part_2 not in months_set_2:
            return False
        return part_1 < 32

    # DD Month
    if re.fullmatch('\d?\d[\s/-][a-zA-Z]+', s):
        part_1, part_2 = re.findall('\w+', s)
        part_1, part_2 = int(part_1), part_2.lower()
        if part_2 not in months_set and part_2 not in months_set_2:
            return False
        return part_1 < 32

    # Month DD
    if re.fullmatch('[a-zA-Z]+[\s/-]\d?\d', s):
        part_1, part_2 = re.findall('\w+', s)
        part_1, part_2 = part_1.lower(), int(part_2)
        if part_1 not in months_set and part_1 not in months_set_2:
            return False
        return part_2 < 32

    # YYYY-MM-DD or MM-DD-YYYY or DD-MM-YYYY
    if re.fullmatch('\d+[/-]\d+[/-]\d+', s):
        part_1, part_2, part_3 = re.findall('\d+', s)
        part_1, part_2, part_3 = int(part_1), int(part_2), int(part_3)
        return (part_2 < 13 and part_3 < 32) or ((part_1 < 32 and part_2 < 13) or (part_1 < 13 and part_2 < 32))

    # Month DD YYYY or Month/DD/YYYY or Month-DD-YYYY
    if re.fullmatch('[a-zA-Z]+[\s/-]\d?\d[\s/-]\d+', s):
        part_1, part_2, part_3 = re.findall('\w+', s)
        part_1, part_2, part_3 = part_1.lower(), int(part_2), int(part_3)
        if part_1 not in months_set and part_1 not in months_set_2:
            return False
        return part_2 < 32

    # Month YYYY
    if re.fullmatch('[a-zA-Z]+[\s-]\d{1,4}', s):
        part_1, part_2 = re.findall('\w+', s)
        part_1, part_2 = part_1.lower(), int(part_2)
        return part_1 in months_set or part_1 in months_set_2

    return False

def date_time_word_finder(s: str):
    s_arr = wordpunct_tokenize(s.lower())
    num_word = len(s_arr)
    reg_datetime = '\b\w+[/-]\w+\b|\b\w+[/-]\w+[/-]\w+\b'
    num_dates = len([s for s in re.findall(reg_datetime, s) if is_date(s)])
    num_terms = len([s for s in s_arr if s in date_time_words])
    return num_terms, num_dates

class DateTimePreproc(IPreprocessor):
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.transform(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return scipy.sparse.csr_matrix(np.array([date_time_word_finder(s) for s in data]))

    def __name__(self):
        return 'date-time counts'

class MultiPreProc(IPreprocessor):

    def __init__(self):
        self.p1 = CountPreprocessor()
        self.p2 = DateTimePreproc()

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        data1 = self.p1.fit_transform(data)
        data2 = self.p2.fit_transform(data)
        return scipy.sparse.hstack((data2, data1))

    def transform(self, data: np.ndarray) -> np.ndarray:
        data1 = self.p1.transform(data)
        data2 = self.p2.transform(data)
        return scipy.sparse.hstack((data2, data1))

    def __name__(self):
        return 'data-time counts with document vectorized with tf-idf'


class ScalerPreProc(IPreprocessor):

    def __init__(self):
        self.p1 = TfIdfPreprocessor()
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        data1 = self.p1.fit_transform(data)
        data2 = self.scaler.fit_transform(data1)
        return data2

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.transform(self.p1.transform(data))

    def __name__(self):
        return 'baseline tf-idf scaled by Standard scaler'