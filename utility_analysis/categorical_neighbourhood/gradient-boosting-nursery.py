from sklearn.model_selection import RandomizedSearchCV
from utils import *
from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood
import numpy as np
from time import time
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit

gamma = 3
n_exp = 10

start = time()

# brast cancer data
breast_cancer, pk = import_dataset('nursery_full')
print(breast_cancer)
# preprocess
breast_cancer = breast_cancer.drop('Id', axis=1)
# separate target from the data
c = len(breast_cancer.columns)
target = breast_cancer.values[:, (c-1)]
le = LabelEncoder()
target = le.fit_transform(target)

breast_cancer = breast_cancer.drop("target", axis=1)
# one-hot encode
breast_cancer = pd.get_dummies(breast_cancer)
# a bit more preprocessing
breast_cancer = breast_cancer.drop(['finance_inconv'], axis=1)
data = breast_cancer.values

# define the model and possible hyperparameters
random_state = 21 # increase every run
n_estimators_range = range(70, 201, 10)
loss_range = ['deviance', 'exponential']
criterion_range = ['friedman_mse', 'mse', 'mae']

# hyperparameter random search
# take the best accuracy from 10-fold cross validation as a benchmark performance
model = GradientBoostingClassifier(random_state=random_state)
#param_dist = dict(criterion=criterion_range, loss=loss_range, n_estimators=n_estimators_range)
#rand_search = RandomizedSearchCV(model, param_dist, cv=10, n_iter=5, scoring="accuracy", random_state=random_state)
#rand_search.fit(data, target)
#best_params = rand_search.best_params_
#print(best_params)
#print(rand_search.best_score_)
#print(rand_search.best_estimator_)
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=random_state)
#scores = cross_val_score(model, data, target, cv=cv)
#print(scores)
#print("TIME " + str(int(time()-start)))
#exit()

results = []
secret_key = 3255
for n in range(n_exp):
    # fingerprint the data
    scheme = CategoricalNeighbourhood(gamma=gamma, xi=2, fingerprint_bit_length=64)

    fp_dataset = scheme.insertion(dataset_name="nursery", buyer_id=1, secret_key=secret_key)
    # same prepocessing as above
    fp_dataset = fp_dataset.drop("Id", axis=1)
    fp_dataset = pd.get_dummies(fp_dataset)
    fp_dataset = fp_dataset.drop(['finance_inconv'], axis=1)
    fp_dataset = fp_dataset.values
    # hyperparameter seach

    model2 = GradientBoostingClassifier(random_state=random_state)
    scores = cross_val_score(model2, fp_dataset, target, cv=cv)
    print(np.mean(scores))
    # note the  best accuracy from the 10-fold
    # repeat
    results.append(np.mean(scores))
    secret_key = secret_key - 3
    print(secret_key)

print(np.mean(results))
print("Time: " + str(int(time()-start)) + " sec.")

# --------- # 0.7798611111111111 # 0.8449845679012346  # 0.9838220169999999
# --------- # decision tree      # logistic regression # gradient boosting # knn
# gamma = 1                                            # 0.9255984224965706
# gamma = 3 |                    # 0.8371504131348129  # 0.963346193415638
# gamma = 5 | 0.7712789981739371 # 0.8406252166787789  # 0.970838477366255
# -----------------------------------------------#
# gamma = 10 | 0.7739561900738382 # 0.8429010038165808 # 0.9774554183813444
# -----------------------------------------------#
# gamma = 20 | 0.7716050642129408 # 0.8439648254200366 # 0.980536694101509
# -----------------------------------------------#
# gamma = 30 | 0.7758340548858322 # 0.844169206134343  # 0.9813134430727023
# ------------------------------------------------