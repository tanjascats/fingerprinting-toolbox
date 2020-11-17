from sklearn.tree import DecisionTreeClassifier
from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

import pandas as pd
import numpy as np


def evaluate_original_bc_dt():
    # EVALUATE ORIGINAL UTILITY
    data = pd.read_csv('datasets/breast_cancer_full.csv', index_col='Id')
    #label_encoder = LabelEncoder()
    #for col in data.columns:
    #    data[col] = label_encoder.fit_transform(data[col])
    target = data['recurrence']
    data = data.drop('recurrence', axis=1)
    data = pd.get_dummies(data)
    data = data.drop(['breast_left', 'irradiat_yes'], axis=1)
    model = DecisionTreeClassifier(random_state=0, max_depth=2, criterion='entropy')
    scores = cross_val_score(model, data, target, cv=10)
    print('original score: ' + str(np.mean(scores)))
    return target


def evaluate_fingerprinted_bc_dt(gamma, target):
    score = []
    for i in range(50):
        # EVALUATE FINGERPRINTED; gamma = 1
        scheme = CategoricalNeighbourhood(gamma=gamma, xi=1, fingerprint_bit_length=8)
        fingerprinted = scheme.insertion('breast_cancer', buyer_id=0, secret_key=333+i)
        # fingerprinted = fingerprinted.drop('recurrence', axis=1)
        fingerprinted = fingerprinted.drop('Id', axis=1)
        fingerprinted = pd.get_dummies(fingerprinted)
        fingerprinted = fingerprinted.drop(['breast_left', 'irradiat_yes'], axis=1)

        model = DecisionTreeClassifier(random_state=0, max_depth=2, criterion='entropy')
        scores = cross_val_score(model, fingerprinted, target, cv=10)
        score.append(scores)
    print('Fingerprinted score, gamma ' + str(gamma) + " " + str(np.mean(score)))


if __name__ == '__main__':
    target = evaluate_original_bc_dt()  # 0.7163
    evaluate_fingerprinted_bc_dt(5, target)   #
