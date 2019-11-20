# this script runs the experiment on data utility of a fingerprinted dataset
# when it is used to build a decision tree model

# the workflow, as well as one run of the experiment is described in jupyter
# notebook Utility Analysis on Machine Learning Performance
import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from utils import *

n_experiments = 1
start = time.time()

for i in range(n_experiments):
    # i divide a random portion of data (20%) and keep it until the test phase
    dataset_name = "german_credit_full"
    german_credit, primary_key = import_dataset(dataset_name)
    # take a random holdout
    holdout = german_credit.sample(frac=0.2, random_state=0)
    german_credit = german_credit.drop(holdout.index, axis=0)

    # one-hot encode the categorical vals but first label-encode the catergorical because OneHot cant't handle them LOL
    label_enc = LabelEncoder()
    categorical_attributes = german_credit.select_dtypes(include='object').columns
    for cat in categorical_attributes:
        german_credit[cat] = label_enc.fit_transform(german_credit[cat])
        holdout[cat] = label_enc.fit_transform(holdout[cat])
    c = len(german_credit.columns)
    data = german_credit.values[:, 1:(c - 1)]
    target = german_credit.values[:, (c - 1)]

    categorical_features_idx = [i - 1 for i in range(len(german_credit.columns)) if
                                german_credit.columns[i] in categorical_attributes]
    encoder = OneHotEncoder(categorical_features=categorical_features_idx)
    data = encoder.fit_transform(data).toarray().astype(np.int)

    model = DecisionTreeClassifier(random_state=0)

    criterion_range = ["gini", "entropy"]
    max_depth_range = range(1, 30)
    param_dist = dict(criterion=criterion_range, max_depth=max_depth_range)
    rand = RandomizedSearchCV(model, param_dist, cv=10, n_iter=20, scoring="accuracy", random_state=0)
    rand.fit(data, target)
    best_params = rand.best_params_

    # train the model
    model = DecisionTreeClassifier(random_state=0, criterion=best_params['criterion'],
                                   max_depth=best_params['max_depth'])
    model.fit(data, target)
    X_test = holdout.values[:, 1:(c - 1)]
    X_test = encoder.fit_transform(X_test).toarray().astype(np.int)
    y_test = holdout.values[:, (c - 1)]

    model.score(X_test, y_test)
    # fingerprinted
    german_credit_fp = read_data_with_target("german_credit", "categorical_neighbourhood", [7, 2], 0)
    # remove the holdout
    german_credit_fp = german_credit_fp.drop(holdout.index, axis=0)
    # preprocess
    # one-hot encode the categorical vals but first label-encode the catergorical because OneHot cant't handle them LOL
    label_enc = LabelEncoder()
    for cat in categorical_attributes:
        german_credit_fp[cat] = label_enc.fit_transform(german_credit_fp[cat])
    data_fp = german_credit_fp.values[:, 1:(c - 1)]
    target_fp = german_credit_fp.values[:, (c - 1)]

    encoder = OneHotEncoder(categorical_features=categorical_features_idx)
    data_fp = encoder.fit_transform(data_fp).toarray().astype(np.int)
    model = DecisionTreeClassifier(random_state=0, criterion=best_params['criterion'],
                                   max_depth=best_params['max_depth'])
    model.fit(data_fp, target_fp)
    model.score(X_test, y_test)

    print(time.time()-start)
