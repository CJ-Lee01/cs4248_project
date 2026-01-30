# diffs
"""
For each label, we want 3 groups:
Correctly classified labels
Labels that were assigned to other types
Other labels that are misclassified as this label
"""
import os
import pathlib

import pandas as pd

"""
for each category, separate them into the above 3 groups
"""

if __name__ == '__main__':
    pass

LABEL_LIST = [
    'DEFAULT ITER:10', 'BIDIRECTIONAL ITER:10', 'LOW_EMBEDDING ITER:10',
    'HIGH_EMBEDDING ITER:10', 'LOW_HIDDEN ITER:10', 'HIGH_HIDDEN ITER:10',
    'tf-idf baseline', 'CountVectorizer', 'date-time counts',
    'data-time counts with document vectorized with tf-idf', 'baseline tf-idf scaled by Standard scaler', 'SVM'
]

LABELS = [1,2,3,4]

df = pd.read_csv('../data/balancedtest.csv')
print(df.keys())

for pred in LABEL_LIST:
    pred_name = pred.split(':')[0].replace(' ', '_')
    os.mkdir(f'LABELS/{pred_name}')
    for label in LABELS:
        os.mkdir(f'LABELS/{pred_name}/{label}')
        true_pos = df['Text'][df[pred] == df['label']][df['label'] == label].sort_values()
        false_pos = df['Text'][df[pred] == df['label']][df['label'] != label].sort_values()
        false_neg = df['Text'][df[pred] != df['label']][df['label'] == label].sort_values()
        true_pos.to_csv(f'LABELS/{pred_name}/{label}/TP', index=False)
        false_pos.to_csv(f'LABELS/{pred_name}/{label}/FP', index=False)
        false_neg.to_csv(f'LABELS/{pred_name}/{label}/FN', index=False)
