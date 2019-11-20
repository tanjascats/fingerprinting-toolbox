# this script runs the experiment on data utility of a fingerprinted dataset
# when it is used to build a decision tree model

# the workflow, as well as one run of the experiment is described in jupyter
# notebook Utility Analysis on Machine Learning Performance
import numpy as np
import time
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from utils import *
from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood

scheme_params = {'gamma': 3, 'xi': 2}
n_experiments = 5  # number of runs withing one single fingerprinted data set
n_fingerprint_experiments = 10  # how many times do I want to fingerprint the data set
fail = 0
score_diff_total = 0
start = time.time()
random.seed(333)

scheme = CategoricalNeighbourhood(gamma=scheme_params['gamma'], xi=scheme_params['xi'])

print("Running " + str(n_experiments) + " experiments.")

dataset_name = "german_credit_full"
german_credit_original, primary_key = import_dataset(dataset_name)
german_credit = german_credit_original.copy()

# one-hot encode the categorical vals but first label-encode the catergorical because OneHot cant't handle them LOL
label_enc = LabelEncoder()
categorical_attributes = german_credit.select_dtypes(include='object').columns
for cat in categorical_attributes:
    german_credit[cat] = label_enc.fit_transform(german_credit[cat])
c = len(german_credit.columns)
data = german_credit.values[:, 1:(c - 1)]
target = german_credit.values[:, (c - 1)]

categorical_features_idx = [i - 1 for i in range(len(german_credit.columns)) if
                            german_credit.columns[i] in categorical_attributes]
encoder = OneHotEncoder(categorical_features=categorical_features_idx)
data = encoder.fit_transform(data).toarray().astype(np.int)

# this should also be in a loop because the randomness of the fingerprint affects the results
for fp_exp in range(n_fingerprint_experiments):

    german_credit_fp = scheme.insertion(dataset_name="german_credit", buyer_id=0, secret_key=random.randint(0, 1000))
    german_credit_fp = add_target(german_credit_fp, "german_credit")

    for cat in categorical_attributes:
        german_credit_fp[cat] = label_enc.fit_transform(german_credit_fp[cat])
    data_fp = german_credit_fp.values[:, 1:(c - 1)]
    target_fp = german_credit_fp.values[:, (c - 1)]
    data_fp = encoder.fit_transform(data_fp).toarray().astype(np.int)

    for exp in range(n_experiments):
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
        try:
            rand_search.fit(data, target)
        except ValueError:
            fail += 1
            continue
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

        score_diff = score_fp - score_original
        score_diff_total += score_diff

score_diff_total /= (n_experiments*n_fingerprint_experiments - fail)
print("Average accuracy difference: " + str(score_diff_total))
print("Time: " + str(int(time.time()-start)) + " seconds.")
