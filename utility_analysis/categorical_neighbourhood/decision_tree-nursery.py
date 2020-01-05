from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from utils import *
from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood
import numpy as np

gamma = 5
n_exp = 1

# nursery data
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
print(breast_cancer.columns)

# a bit more preprocessing
breast_cancer = breast_cancer.drop('finance_inconv', axis=1)
data = breast_cancer.values

# define the model and possible hyperparameters
random_state = 25 # increase every run
criterion_range = ["gini", "entropy"]
max_depth_range = range(1, 30)

# hyperparameter random search
# take the best accuracy from 10-fold cross validation as a benchmark performance
model = DecisionTreeClassifier(random_state=random_state)
param_dist = dict(criterion=criterion_range, max_depth=max_depth_range)
rand_search = RandomizedSearchCV(model, param_dist, cv=10, n_iter=40, scoring="accuracy", random_state=random_state)
rand_search.fit(data, target)
best_params = rand_search.best_params_
print(best_params)
print(rand_search.best_score_)
print(rand_search.best_estimator_)

results = []
for n in range(n_exp):
    # fingerprint the data
    scheme = CategoricalNeighbourhood(gamma=gamma, xi=2, fingerprint_bit_length=64)
    secret_key = 3255  # increase every run
    fp_dataset = scheme.insertion(dataset_name="nursery", buyer_id=1, secret_key=secret_key)
    # same prepocessing as above
    fp_dataset = fp_dataset.drop("Id", axis=1)
    fp_dataset = pd.get_dummies(fp_dataset)
    fp_dataset = fp_dataset.drop('finance_inconv', axis=1)
    fp_dataset = fp_dataset.values
    # hyperparameter seach

    model2 = DecisionTreeClassifier(random_state=secret_key)
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

# --------- # 0.7798611111111111 #
# --------- # decision tree      # gradient boosting # knn
# gamma = 1 |
# -----------------------------------------------#
# gamma = 2 |
# -----------------------------------------------#
# gamma = 3 |
# -----------------------------------------------#
# gamma = 5 |
# ------------------------------------------------