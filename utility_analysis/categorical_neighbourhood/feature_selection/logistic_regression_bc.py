from sklearn.model_selection import RandomizedSearchCV
from utils import *
from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood
import numpy as np
from time import time
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

gamma = 5
n_exp = 20
scoring = "accuracy"

start = time()

# brast cancer data
breast_cancer, pk = import_dataset('breast_cancer_full')

# preprocess
breast_cancer = breast_cancer.drop('Id', axis=1)
# separate target from the data
c = len(breast_cancer.columns)
target = breast_cancer.values[:, (c-1)]
breast_cancer = breast_cancer.drop("recurrence", axis=1)
print(breast_cancer.columns)
n_features = len(breast_cancer.columns)
# one-hot encode
breast_cancer = pd.get_dummies(breast_cancer)
# a bit more preprocessing
breast_cancer = breast_cancer.drop(['breast_left', 'irradiat_yes'], axis=1)
data = breast_cancer.values


# define the model and possible hyperparameters
random_state = 13 # increase every run

secret_key = 823  # change every run
results = []
for n in range(n_exp):
    result = []
    # fingerprint the data
    scheme = CategoricalNeighbourhood(gamma=gamma, xi=2, fingerprint_bit_length=8)
    fp_dataset = scheme.insertion(dataset_name="breast_cancer", buyer_id=1, secret_key=secret_key)
    # same prepocessing as above
    fp_dataset = fp_dataset.drop("Id", axis=1)
    fp_dataset_values = fp_dataset.values
    # dummmies from here
    fp_dataset_one_hot = pd.get_dummies(fp_dataset)
    fp_dataset_one_hot = fp_dataset_one_hot.drop(['breast_left', 'irradiat_yes'], axis=1)
    fp_dataset_one_hot = fp_dataset_one_hot.values
    # hyperparameter seach

    model2 = LogisticRegression(random_state=random_state, C=90, solver='saga')
    scores = cross_val_score(model2, fp_dataset_one_hot, target, cv=10, scoring=scoring)
    print(np.mean(scores))
    # note the  best accuracy from the 10-fold
    # repeat
    result.append(np.mean(scores))
    secret_key = secret_key - 3

    features_left = n_features
    for feature in range(n_features-1):
        # drop one feature
        #label encode
        le = LabelEncoder()
        encoded = fp_dataset.apply(le.fit_transform)
        selection = SelectKBest(chi2, k=features_left-1)
        X_new = selection.fit_transform(encoded, target)
        # moram napravit dataframe od X_new
        new_features = encoded.columns[selection.get_support()]
        print(new_features)
        X_new = pd.DataFrame(data=X_new[:,:], columns=new_features)
        features_left -= 1

        # evaluate performance
        X_new_one_hot = pd.get_dummies(X_new)
        X_new_one_hot = X_new_one_hot.values
        model = LogisticRegression(random_state=random_state, C=90, solver='saga')
        scores = cross_val_score(model, X_new_one_hot, target, cv=10, scoring=scoring)
        print(np.mean(scores))
        result.append(np.mean(scores))
    results.append(result)
print(results)
print("Mean values:")
print("Scoring: " + scoring)
for i in range(len(results[0])):
    print(np.mean([r[i] for r in results]))

print("Time: " + str(int(time()-start)) + " sec.")

# solver:saga, C=90