# this script runs the experiment on data utility of a fingerprinted dataset
# when it is used to build a decision tree model

# the workflow, as well as one run of the experiment is described in jupyter
# notebook Utility Analysis on Machine Learning Performance
import pandas as pd
import numpy as np
import time
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from utils import *

n_experiments = 5
start = time.time()
random.seed(333)

# things i have to do only once
  # load the dataset
  # load the fingerprinted
  # label-encode the original
  # hot-encode the original
  # label-encode the fingerprinted
  # hot-encode the fingeprinted

dataset_name = "german_credit_full"
german_credit_original, primary_key = import_dataset(dataset_name)
german_credit = german_credit_original.copy()

# this should also be in a loop because the randomness of the fingerprint affects the results
# todo: if the file doesnt exist; invoke the fingerprinting func
german_credit_fp = read_data_with_target("german_credit", "categorical_neighbourhood", [1, 2], 0)

# one-hot encode the categorical vals but first label-encode the catergorical because OneHot cant't handle them LOL
label_enc = LabelEncoder()
categorical_attributes = german_credit.select_dtypes(include='object').columns
for cat in categorical_attributes:
    german_credit[cat] = label_enc.fit_transform(german_credit[cat])
    german_credit_fp[cat] = label_enc.fit_transform(german_credit_fp[cat])
c = len(german_credit.columns)
data = german_credit.values[:, 1:(c - 1)]
data_fp = german_credit_fp.values[:, 1:(c - 1)]
target = german_credit.values[:, (c - 1)]
target_fp = german_credit_fp.values[:, (c - 1)]

categorical_features_idx = [i - 1 for i in range(len(german_credit.columns)) if
                            german_credit.columns[i] in categorical_attributes]
encoder = OneHotEncoder(categorical_features=categorical_features_idx)
data = encoder.fit_transform(data).toarray().astype(np.int)
data_fp = encoder.fit_transform(data_fp).toarray().astype(np.int)

for i in range(n_experiments):
    # take a random holdout 20%
    holdout_idx = random.sample([i for i in range(len(data))], int(0.2*len(data)))
    holdout = data[holdout_idx]
    holdout_target = target[holdout_idx]

    # data and target of the remaining 80%
    data_idx = [i for i in range(len(data)) if i not in holdout_idx]
    data = data[data_idx]
    target = target[data_idx]

    # initiate a model for hyperparameter search
    model = DecisionTreeClassifier(random_state=0)

    criterion_range = ["gini", "entropy"]
    max_depth_range = range(1, 30)
    param_dist = dict(criterion=criterion_range, max_depth=max_depth_range)
    rand_search = RandomizedSearchCV(model, param_dist, cv=10, n_iter=20, scoring="accuracy", random_state=0)
    rand_search.fit(data, target)
    best_params = rand_search.best_params_

    # train the model with the best found hyperparameters on original data
    model = DecisionTreeClassifier(random_state=0, criterion=best_params['criterion'],
                                   max_depth=best_params['max_depth'])
    model.fit(data, target)
    score_original = model.score(holdout, holdout_target)

    # fingerprinted
    # remove the holdout
    data_fp = data_fp[data_idx]
    target_fp = target_fp[data_idx]

    # define and train a model on the fingerprinted data
    model = DecisionTreeClassifier(random_state=0, criterion=best_params['criterion'],
                                   max_depth=best_params['max_depth'])
    model.fit(data_fp, target_fp)
    score_fp = model.score(holdout, holdout_target)

    print(score_fp-score_original)
    print(time.time()-start)
