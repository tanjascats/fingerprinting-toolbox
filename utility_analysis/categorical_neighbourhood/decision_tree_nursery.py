from sklearn.model_selection import RandomizedSearchCV
from utils import *
from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood
import numpy as np
from time import time
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import itertools
from pprint import pprint
import matplotlib.pyplot as plt

gamma = 30
n_exp = 10
#remove = 2

start = time()

# brast cancer data
breast_cancer, pk = import_dataset('nursery_full')
print(breast_cancer)
# preprocess
breast_cancer = breast_cancer.drop('Id', axis=1)
# separate target from the data
c = len(breast_cancer.columns)
target = breast_cancer.values[:, (c-1)]
breast_cancer = breast_cancer.drop("target", axis=1)

#combinations = list(itertools.combinations(breast_cancer.columns, remove))
#print(combinations)

# one-hot encode
breast_cancer = pd.get_dummies(breast_cancer)
# a bit more preprocessing
breast_cancer = breast_cancer.drop('finance_inconv', axis=1)
data = breast_cancer.values

# define the model and possible hyperparameters
random_state = 18 # increase every run
criterion_range = ["gini", "entropy"]
max_depth_range = range(1, 30)

# hyperparameter random search
# take the best accuracy from 10-fold cross validation as a benchmark performance
model = DecisionTreeClassifier(random_state=random_state)
param_dist = dict(criterion=criterion_range, max_depth=max_depth_range)
rand_search = RandomizedSearchCV(model, param_dist, cv=10, n_iter=15, scoring="accuracy", random_state=random_state)
rand_search.fit(data, target)
best_params = rand_search.best_params_
print(best_params)
print(rand_search.best_score_)
print(rand_search.best_estimator_)

secret_key = 3285  # increase every run
results = {'full': []}
for n in range(n_exp):
    # fingerprint the data
    scheme = CategoricalNeighbourhood(gamma=gamma, xi=2, fingerprint_bit_length=8)
    fp_dataset = scheme.insertion(dataset_name="nursery", buyer_id=1, secret_key=secret_key)
    # same prepocessing as above
    fp_dataset = fp_dataset.drop("Id", axis=1)
    fp_dataset_dummies = pd.get_dummies(fp_dataset)
    fp_dataset_dummies = fp_dataset_dummies.drop(['finance_inconv'], axis=1)
    # hyperparameter seach

    model2 = DecisionTreeClassifier(random_state=random_state, criterion=best_params['criterion'],
                                    max_depth=best_params['max_depth'])
    scores = cross_val_score(model2, fp_dataset_dummies.values, target, cv=10)
    print(np.mean(scores))
    # note the  best accuracy from the 10-fold
    # repeat
    results['full'].append(np.mean(scores))
    secret_key = secret_key - 3
'''
    # calculate the accuracy of reduced dataset
    for attribute_combination in combinations:
        if attribute_combination not in results:
            results[attribute_combination] = []
        # feature selected dataset
        fs_dataset = fp_dataset
        for att in attribute_combination:
            fs_dataset = fs_dataset.drop(att, axis=1)
        fs_dataset = pd.get_dummies(fs_dataset)
        if 'finance_inconv' in fs_dataset.columns:
            fs_dataset = fs_dataset.drop(['finance_inconv'], axis=1)
        model3 = DecisionTreeClassifier(random_state=random_state, criterion=best_params['criterion'],
                                        max_depth=best_params['max_depth'])
        scores = cross_val_score(model3, fs_dataset.values, target, cv=10)
        results[attribute_combination].append(np.mean(scores))
'''


#print(np.mean(results))
pprint(results)
print('Full: ' + str(np.mean(results['full'])))
results_avg = [np.mean(results[r]) for r in results]
results_avg.remove(np.mean(results['full']))
print(sorted(results_avg))
print("Time: " + str(int(time()-start)) + " sec.")

# --------- # 0.7798611111111111 #
# --------- # decision tree      # gradient boosting # knn
# gamma = 1 | 0.7614499284525335 #
# -----------------------------------------------
# gamma = 3 |
# -----------------------------------------------#
# gamma = 5 | 0.7712789981739371
# -----------------------------------------------#
# gamma = 10 |
# -----------------------------------------------#
# gamma = 20 |
# -----------------------------------------------#
# gamma = 30 |
# ------------------------------------------------
#
