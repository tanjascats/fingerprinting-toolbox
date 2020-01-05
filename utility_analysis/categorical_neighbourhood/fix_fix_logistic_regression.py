from sklearn.model_selection import RandomizedSearchCV
from utils import *
from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood
import numpy as np
from time import time
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

gamma = 5
n_exp = 100

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
random_state = 25 # increase every run
solver_range = ["liblinear", "newton-cg", "lbfgs", "saga"]
C_range = range(10, 101, 10)

# hyperparameter random search
# take the best accuracy from 10-fold cross validation as a benchmark performance
model = LogisticRegression(random_state=random_state)
param_dist = dict(solver=solver_range, C=C_range)
rand_search = RandomizedSearchCV(model, param_dist, cv=10, n_iter=15, scoring="accuracy", random_state=random_state)
rand_search.fit(data, target)
best_params = rand_search.best_params_
print(best_params)
print(rand_search.best_score_)
print(rand_search.best_estimator_)
exit()
secret_key = 3255  # change every run
results = []
for n in range(n_exp):
    # fingerprint the data
    scheme = CategoricalNeighbourhood(gamma=gamma, xi=2, fingerprint_bit_length=8)
    fp_dataset = scheme.insertion(dataset_name="breast_cancer", buyer_id=1, secret_key=secret_key)
    # same prepocessing as above
    fp_dataset = fp_dataset.drop("Id", axis=1)
    fp_dataset = pd.get_dummies(fp_dataset)
    fp_dataset = fp_dataset.drop(['breast_left', 'irradiat_yes'], axis=1)
    fp_dataset = fp_dataset.values
    # hyperparameter seach

    model2 = LogisticRegression(random_state=random_state, C=best_params['C'], solver=best_params['solver'])
    scores = cross_val_score(model2, fp_dataset, target, cv=10)
    print(np.mean(scores))
    # note the  best accuracy from the 10-fold
    # repeat
    results.append(np.mean(scores))
    secret_key = secret_key - 3

print(np.mean(results))
print("Time: " + str(int(time()-start)) + " sec.")

# --------- # 0.7167832167832168 # 0.6678321678321678  # 0.6783216783216783 # 0.7517482517482518
# --------------------------------------------------------------------------------------------
# --------- # decision tree      # logistic regression # gradient boosting  # knn
# gamma = 1 | 0.6853146853146851 # 0.6663658456486042  #                   #
# ----------------------------------------------------------------------------------------------#
# gamma = 2 | 0.6993006993006995 # 0.6670189655172414
# ----------------------------------------------------------------------------------------------#
# gamma = 3 | 0.6643356643356643 # 0.6608391608391608  #
# ----------------------------------------------------------------------------------------------#
# gamma = 5 | 0.6993006993006995 # 0.6643120689655173
# ------------------------------------------------
# ------------------------------------------------
# ------------------------------------------------
# gamma = 1 |        ?                     ?                     ?          # 0.7382266009852217
# gamma = 2          ?                     ?                      ?         # 0.7308374384236453
# gamma = 3          ?                     ?                      ?         # 0.7381034482758622
# gamma = 5          ?                     ?                      ?         # 0.7488177339901477
#