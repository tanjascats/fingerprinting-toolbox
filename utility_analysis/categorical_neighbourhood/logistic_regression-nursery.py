from sklearn.model_selection import RandomizedSearchCV
from utils import *
from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood
import numpy as np
from time import time
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

gamma = 3
n_exp = 20

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
# one-hot encode
breast_cancer = pd.get_dummies(breast_cancer)
# a bit more preprocessing
breast_cancer = breast_cancer.drop(['finance_inconv'], axis=1)
data = breast_cancer.values

# define the model and possible hyperparameters
random_state = 25 # increase every run
solver_range = ["liblinear", "newton-cg", "lbfgs", "saga"]
C_range = range(10, 101, 10)

# hyperparameter random search
# take the best accuracy from 10-fold cross validation as a benchmark performance
#model = LogisticRegression(random_state=random_state)
#param_dist = dict(solver=solver_range, C=C_range)
#rand_search = RandomizedSearchCV(model, param_dist, cv=10, n_iter=15, scoring="accuracy", random_state=random_state)
#rand_search.fit(data, target)
#best_params = rand_search.best_params_
#print(best_params)
#print(rand_search.best_score_)
#print(rand_search.best_estimator_)

secret_key = 3255  # change every run
results = []
for n in range(n_exp):
    # fingerprint the data
    scheme = CategoricalNeighbourhood(gamma=gamma, xi=2, fingerprint_bit_length=8)
    fp_dataset = scheme.insertion(dataset_name="nursery", buyer_id=1, secret_key=secret_key)
    # same prepocessing as above
    fp_dataset = fp_dataset.drop("Id", axis=1)
    fp_dataset = pd.get_dummies(fp_dataset)
    fp_dataset = fp_dataset.drop(['finance_inconv'], axis=1)
    fp_dataset = fp_dataset.values
    # hyperparameter seach

    model2 = LogisticRegression(random_state=random_state, C=20, solver='lbfgs')
    scores = cross_val_score(model2, fp_dataset, target, cv=10)
    print(np.mean(scores))
    # note the  best accuracy from the 10-fold
    # repeat
    results.append(np.mean(scores))
    secret_key = secret_key - 3

print(np.mean(results))
print("Time: " + str(int(time()-start)) + " sec.")

# --------- # 0.7798611111111111 # 0.8449845679012346
# --------- # decision tree      # logistic regression # gradient boosting # knn
# gamma = 5 | 0.7712789981739371
# -----------------------------------------------#
# gamma = 10 | 0.7739561900738382
# -----------------------------------------------#
# gamma = 20 | 0.7716050642129408
# -----------------------------------------------#
# gamma = 30 | 0.7758340548858322
# ------------------------------------------------
