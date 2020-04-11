from sklearn.model_selection import RandomizedSearchCV
from utils import *
from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood
import numpy as np
from time import time
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import itertools
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import random

import matplotlib.pyplot as plt

gamma = 5  # todo: all gammas
n_exp = 15  # todo 100 or 15
# how many attributes to remove in a vertical attack
# todo: list of all possible values
removes = [i for i in range(1, 8)]

f = open("robustness_vs_utility/log/nursery" +
         str(datetime.today().date()) + "-" +
         str(datetime.today().time()).split(":")[0] + "." +
         str(datetime.today().time()).split(":")[1] + "." +
         str(datetime.today().time()).split(":")[2] + ".txt",
         "a+")

start = time()

# nursery dataset
dataset, pk = import_dataset('nursery_full')
# preprocess
dataset = dataset.drop('Id', axis=1)
# separate target from the data
c = len(dataset.columns)
target = dataset.values[:, (c-1)]
nursery = dataset.drop("target", axis=1)

combinations = dict()
for remove in removes:
    combinations[remove] = list(itertools.combinations(nursery.columns, remove))
print(combinations)

# one-hot encode
nursery = pd.get_dummies(nursery)
# a bit more preprocessing
nursery = nursery.drop('finance_inconv', axis=1)
data = nursery.values

# define the model and possible hyperparameters
random_state = 18 # increase every run
criterion_range = ["gini", "entropy"]
max_depth_range = range(1, 30)

# hyperparameter random search
# take the best accuracy from 10-fold cross validation as a benchmark performance
# model = DecisionTreeClassifier(random_state=random_state)
# param_dist = dict(criterion=criterion_range, max_depth=max_depth_range)
# rand_search = RandomizedSearchCV(model, param_dist, cv=10, n_iter=15, scoring="accuracy", random_state=random_state)
# rand_search.fit(data, target)
# best_params = rand_search.best_params_
# print(best_params)
# print(rand_search.best_score_)
# print(rand_search.best_estimator_)
print("k-NN", f)
print("\t-best parameters: {'n_neighbors': 8, 'algorithm': 'kd_tree'}")
print("\t-best accuracy score: 0.7689")

secret_key = 3285  # increase every run
results = {'full': []}
means = dict()
maxes = dict()
mins = dict()
for n in range(n_exp):
    # fingerprint the data
    scheme = CategoricalNeighbourhood(gamma=gamma, xi=2, fingerprint_bit_length=8)
    fp_dataset = scheme.insertion(dataset_name="nursery", buyer_id=1, secret_key=secret_key)
    # same prepocessing as above
    fp_dataset = fp_dataset.drop("Id", axis=1)
    fp_dataset_dummies = pd.get_dummies(fp_dataset)
    fp_dataset_dummies = fp_dataset_dummies.drop(['finance_inconv'], axis=1)
    # hyperparameter seach

    model2 = KNeighborsClassifier(algorithm='kd_tree', n_neighbors=8)
    scores = cross_val_score(model2, fp_dataset_dummies.values, target, cv=10)
    # note the  best accuracy from the 10-fold
    # repeat
    results['full'].append(np.mean(scores))

    # calculate the accuracy of reduced dataset
    # iterate through every number of removed attrs
    for remove in removes:
        if len(combinations[remove]) > 40:
            sample_size = int(0.8*len(combinations[remove]))
        else:
            sample_size = len(combinations[remove])
        combinations_to_consider = random.sample(combinations[remove], sample_size)
        for attribute_combination in combinations_to_consider:
            if attribute_combination not in results:
                results[attribute_combination] = []
            # feature selected dataset
            fs_dataset = fp_dataset
            for att in attribute_combination:
                fs_dataset = fs_dataset.drop(att, axis=1)
            fs_dataset = pd.get_dummies(fs_dataset)
            if 'finance_inconv' in fs_dataset.columns:
                fs_dataset = fs_dataset.drop(['finance_inconv'], axis=1)
            model3 = KNeighborsClassifier(n_neighbors=8, algorithm='kd_tree')
            scores = cross_val_score(model3, fs_dataset.values, target, cv=10)
            # results for each possible value combination
            results[attribute_combination].append(np.mean(scores))

        # pprint("Removed attribtues: " + str(remove), f)
        # pprint('Accuracies of each combination of removed attributes: ', f)
        # pprint(results, f)
        # pprint('Full: ' + str(np.mean(results['full'])))
        # results
        results_avg = [np.mean(results[r]) for r in results]
        results_avg.remove(np.mean(results['full']))
        # pprint("Average results (experiments for each attribute combination averaged out):", f)
        #pprint(sorted(results_avg), f)
        #pprint('Avg: ' + str(np.mean(results_avg)), f)

        if remove not in means:
            means[remove] = []
        # record the average results for different combos of removed attributes
        means[remove].append(np.mean(results_avg))
        #pprint('Max: ' + str(np.max(results_avg)), f)
        if remove not in maxes:
            maxes[remove] = []
        # record the maximum obtained accuracies for removing certain number of attributes
        maxes[remove].append(np.max(results_avg))
        #pprint('Min: ' + str(np.min(results_avg)), f)
        if remove not in mins:
            mins[remove] = []
        # record the minimums
        mins[remove].append(np.min(results_avg))

    secret_key = secret_key - 3

#print(np.mean(results))
print("Time: " + str(int(time() - start)) + " sec.")
pprint("Particular results: ", f)
pprint(results, f)

pprint("Avgs: " + str(means), f)
pprint("Maxes:" + str(maxes), f)
pprint("Mins: " + str(mins), f)
pprint("gamma: " + str(gamma), f)
pprint('k-NN', f)
f.close()
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

