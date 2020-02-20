from sklearn.model_selection import RandomizedSearchCV
from utils import *
from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood
import numpy as np
from time import time
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

gamma = 1
n_exp = 10

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
best_params = {'n_estimators': 200, 'loss': 'exponential', 'criterion': 'mae'}
print(best_params)

results = []
secret_key = 3255
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

    model2 = GradientBoostingClassifier(random_state=random_state, n_estimators=200, loss='exponential',
                                        criterion='mae')
    scores = cross_val_score(model2, fp_dataset, target, cv=10)
    print(np.mean(scores))
    # note the  best accuracy from the 10-fold
    # repeat
    results.append(np.mean(scores))
    secret_key = secret_key - 3
    print(secret_key)

print(np.mean(results))
print("Time: " + str(int(time()-start)) + " sec.")

# --------- # 0.7167832167832168 # 0.6678321678321678  # 0.6818181818181818 # 0.7517482517482518
# --------------------------------------------------------------------------------------------
# --------- # decision tree      # logistic regression # gradient boosting  # knn
# gamma = 1 | 0.6853146853146851 #                     # 0.6704178981937603 # 0.7331096880131364
# ----------------------------------------------------------------------------------------------#
# gamma = 2 | 0.6993006993006995                       # 0.6778477011494253 # 0.738843472906404
# ----------------------------------------------------------------------------------------------#
# gamma = 3 | 0.6643356643356643 # 0.6608391608391608  # 0.675712643678161  # 0.7399804187192118
# ----------------------------------------------------------------------------------------------#
# gamma = 5 | 0.6993006993006995                       # 0.6788563218390805 # 0.7440073481116585
# ------------------------------------------------
# ------------------------------------------------
# ------------------------------------------------
# gamma = 1 |        ?                     ?                     ?          # 0.7382266009852217
# gamma = 2          ?                     ?                      ?         # -
# gamma = 3          ?                     ?                      ?         # -
# gamma = 5          ?                     ?                      ?         # 0.7440073481116585
#