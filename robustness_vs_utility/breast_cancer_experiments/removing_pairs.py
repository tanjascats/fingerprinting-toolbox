# import data
# choose a classifier
# calculate (or retrieve) the utility of the original data set
# fingerprint -> choose 1 gamma
# calculate (or retrieve) the average utility of the fingerprinted data set (full)
# remove 2 columns; calculate the average utility
#   -> repeat n times
#        -> repeat for each pair of attributes (or a subset if there is too much of them)
# plot how the utility changes by removing which attribute

# todo: focus on the robustness after this

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood
from sklearn.model_selection import cross_val_score
from pprint import pprint
import itertools
from datetime import datetime


def demo_exp():
    data = pd.read_csv('datasets/breast_cancer_full.csv', index_col='Id')
    target = data.values[:, (len(data.columns) - 1)]
    # retrieving performance from the previous runs
    accuracy_original = 0.7168  # for decision tree: max_depth = 2, criterion = 'entropy'
    gamma = 5
    accuracy_fingerprinted_full = 0.6994

    # try removing each attribute and record the results
    score = dict()
    n_exp = 100
    random_state = 25
    attributes = list(itertools.combinations(data.columns[:-1], 2))
    attributes = [list(pair) for pair in attributes]
    for attr_pair in attributes:
        # calculate utility of attacked fingerprinted data set
        secret_key = 3255  # increase every run
        for n in range(n_exp):
            # fingerprint the data
            scheme = CategoricalNeighbourhood(gamma=gamma, xi=2, fingerprint_bit_length=8)
            fp_dataset = scheme.insertion(dataset_name="breast_cancer", buyer_id=1, secret_key=secret_key)
            fp_dataset = fp_dataset.drop("Id", axis=1)

            # attack
            fp_dataset_attack = fp_dataset.drop(attr_pair, axis=1)
            fp_dataset_attack = pd.get_dummies(fp_dataset_attack)
            if 'breast_left' in fp_dataset_attack:
                fp_dataset_attack = fp_dataset_attack.drop(['breast_left'], axis=1)
            if 'irradiat_yes' in fp_dataset_attack:
                fp_dataset_attack = fp_dataset_attack.drop(['irradiat_yes'], axis=1)

            model2 = DecisionTreeClassifier(random_state=random_state, criterion='entropy', max_depth=2)
            scores = cross_val_score(model2, fp_dataset_attack.values, target, cv=10)
            if str(attr_pair) not in score:
                score[str(attr_pair)] = []
            score[str(attr_pair)].append(np.mean(scores))
            secret_key = secret_key - 3
    print("Fingerprinted full: " + str(accuracy_fingerprinted_full))
    score['full'] = [0.7293225779967158]  # for 100 experiments (from demo_exp.py / results in log_refined)
    pprint(score)
    pprint("Averages")
    pprint({i: np.mean(score[i]) for i in score})
    f = open("robustness_vs_utility/breast_cancer_experiments/log.txt", "a+")
    pprint(score, f)
    pprint("Averages", f)
    pprint({i: np.mean(score[i]) for i in score}, f)
    pprint("Timestamp: " + str(datetime.today()), f)
    pprint("------------------------------------------\n------------------------------------------", f)
    f.close()

    with open("robustness_vs_utility/breast_cancer_experiments/log_refined.txt", "a+") as log_refined:
        log_refined.write("Removing pairs of attributes\n")
        log_refined.write("Decision Tree\n")
        log_refined.write("gamma = " + str(gamma) + "\n")
        log_refined.write("number of experiments = " + str(n_exp) + "\n")
        log_refined.write("Averages:\n")
        keys = map(str, {i: np.mean(score[i]) for i in score})
        items = (key + ": " + str(np.mean(score[key])) for key in keys)
        line = ",\n".join(items)
        log_refined.write(line)
        log_refined.write("\nTimestamp: " + str(datetime.today()) + "\n")
        log_refined.write("------------------------------------------\n------------------------------------------")


if __name__ == '__main__':
    demo_exp()
