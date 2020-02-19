from sklearn.model_selection import RandomizedSearchCV
from utils import *
from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood
import numpy as np
from time import time
from sklearn.ensemble import GradientBoostingClassifier

gamma = 1
n_exp = 5

start = time()

# brast cancer data
breast_cancer, pk = import_dataset('breast_cancer_full')
print(breast_cancer)
# preprocess
breast_cancer = breast_cancer.drop('Id', axis=1)
# separate target from the data
c = len(breast_cancer.columns)
target = breast_cancer.values[:, (c-1)]
breast_cancer = breast_cancer.drop("recurrence", axis=1)
# one-hot encode
breast_cancer = pd.get_dummies(breast_cancer)
# a bit more preprocessing
breast_cancer = breast_cancer.drop(['breast_left', 'irradiat_yes'], axis=1)
data = breast_cancer.values

# define the model and possible hyperparameters
random_state = 37 # increase every run
n_estimators_range = range(70, 201, 10)
loss_range = ['deviance', 'exponential']
criterion_range = ['friedman_mse', 'mse', 'mae']

# hyperparameter random search
# take the best accuracy from 10-fold cross validation as a benchmark performance
model = GradientBoostingClassifier(random_state=random_state)
param_dist = dict(criterion=criterion_range, loss=loss_range, n_estimators=n_estimators_range)
rand_search = RandomizedSearchCV(model, param_dist, cv=10, n_iter=15, scoring="accuracy", random_state=random_state)
rand_search.fit(data, target)
best_params = rand_search.best_params_
print(best_params)
print(rand_search.best_score_)
print(rand_search.best_estimator_)
print("TIME " + str(int(time()-start)))
exit()

results = []
for n in range(n_exp):
    # fingerprint the data
    scheme = CategoricalNeighbourhood(gamma=gamma, xi=2, fingerprint_bit_length=8)
    secret_key = 3255  # increase every run
    fp_dataset = scheme.insertion(dataset_name="breast_cancer", buyer_id=1, secret_key=secret_key)
    # same prepocessing as above
    fp_dataset = fp_dataset.drop("Id", axis=1)
    fp_dataset = pd.get_dummies(fp_dataset)
    fp_dataset = fp_dataset.drop(['breast_left', 'irradiat_yes'], axis=1)
    fp_dataset = fp_dataset.values
    # hyperparameter seach

    model2 = GradientBoostingClassifier(random_state=secret_key)
    rand_search2 = RandomizedSearchCV(model, param_dist, cv=10, n_iter=10, scoring="accuracy", random_state=random_state)
    rand_search2.fit(fp_dataset, target)
    best_params_fp = rand_search2.best_params_
    best_score_fp = rand_search2.best_score_
    print(best_params_fp)
    print(rand_search2.best_score_)
    # note the  best accuracy from the 10-fold
    # repeat
    results.append(best_score_fp)
    secret_key = secret_key - 3

print(np.mean(results))
print("Time: " + str(int(time()-start)) + " sec.")

# --------- # 0.7167832167832168 # 0.6678321678321678  #
# ------------------------------------------------------
# --------- # decision tree      # logistic regression #
# gamma = 1 | 0.6853146853146851 #
# -----------------------------------------------#
# gamma = 2 | 0.6993006993006995
# -----------------------------------------------#
# gamma = 3 | 0.6643356643356643
# -----------------------------------------------#
# gamma = 5 | 0.6993006993006995
# ------------------------------------------------