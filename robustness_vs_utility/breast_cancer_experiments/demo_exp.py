# import data
# choose a classifier
# calculate (or retrieve) the utility of the original data set
# fingerprint -> choose 1 gamma
# calculate (or retrieve) the average utility of the fingerprinted data set (full)
# remove 1 column; calculate the average utility
#   -> repeat n times
#        -> repeat for each attribute
# plot how the utility changes by removing which attribute

# todo: focus on the robustness after this

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood
from sklearn.model_selection import cross_val_score
from pprint import pprint
import random
import itertools


# assuming all rows are present
def fingerprint_cross_val_score(model, data, fp_data, target, cv=10):
    # shuffle data randomly
    shuffle = [i for i in range(len(data))]
    random.shuffle(shuffle)
    # split the data into k groups
    group_size = int(len(data)/cv)
    k_groups = []
    for i in range(cv):
        start_index = i*group_size
        end_index = (i+1)*group_size
        k_groups.append(shuffle[start_index:end_index])

    # for each unique group
    scores = []
    for i in range(cv):
    # # take a group as a holdout (test) from original data
        holdout_set = data[k_groups[i]]
    # # take the remaining groups as training set -> fingerprint the training set
        training_set_idx = shuffle.copy()
        for index in k_groups[i]:
            training_set_idx.remove(index)
        training_set = fp_data[training_set_idx]
    # # fit the model on training set and evaluate on the test set
        model.fit(training_set, target[training_set_idx])
    # # retain the evaluation score and discard the model
        scores.append(model.score(holdout_set, target[k_groups[i]]))
    # summarize the skill of the model using the sample of model evaluation scores
    return scores


def demo_exp():
    data = pd.read_csv('datasets/breast_cancer_full.csv', index_col='Id')
    target = data.values[:, (len(data.columns) - 1)]
    data = data.drop("recurrence", axis=1)
    # one-hot encode
    data_dumm = pd.get_dummies(data)
    # a bit more preprocessing
    data_dumm = data_dumm.drop(['breast_left', 'irradiat_yes'], axis=1)

    # retrieving performance from the previous runs
    accuracy_original = 0.7168  # for decision tree: max_depth = 2, criterion = 'entropy'
    gamma = 3
    accuracy_fingerprinted_full = 0.6994

    # try removing each attribute and record the results
    score = dict()
    score_realistic = dict()
    n_exp = 100
    random_state = 25
    attributes = list(data.columns)
    attributes.append('full')
    for attr in attributes:
        # calculate utility of attacked fingerprinted data set
        secret_key = 3255  # increase every run
        for n in range(n_exp):
            # fingerprint the data
            scheme = CategoricalNeighbourhood(gamma=gamma, xi=2, fingerprint_bit_length=8)
            fp_dataset = scheme.insertion(dataset_name="breast_cancer", buyer_id=1, secret_key=secret_key)
            fp_dataset = fp_dataset.drop("Id", axis=1)

            if attr is 'full':
                fp_dataset_dumm = pd.get_dummies(fp_dataset)
                fp_dataset_dumm = fp_dataset_dumm.drop(['breast_left', 'irradiat_yes'], axis=1)
                data_dumm_preprocessed = data_dumm

                if len(data_dumm_preprocessed.columns.difference(fp_dataset_dumm.columns)) != 0:
                    data_dumm_preprocessed = data_dumm_preprocessed.drop(data_dumm_preprocessed.columns.difference(
                        fp_dataset_dumm.columns), axis=1)
                if len(fp_dataset_dumm.columns.difference(data_dumm_preprocessed.columns)) != 0:
                    fp_dataset_dumm = fp_dataset_dumm.drop(fp_dataset_dumm.columns.difference(
                        data_dumm_preprocessed.columns), axis=1)

                model = DecisionTreeClassifier(random_state=random_state, criterion='entropy', max_depth=2)
                # score on fingerprinted test data
                scores = cross_val_score(model, fp_dataset_dumm.values, target, cv=10)
                # todo: score on original data
                scores_realistic = fingerprint_cross_val_score(model, data_dumm_preprocessed.values, fp_dataset_dumm.values, target, cv=10)
                if 'full' not in score:
                    score['full'] = []
                score['full'].append(np.mean(scores))
                if 'full' not in score_realistic:
                    score_realistic['full'] = []
                score_realistic['full'].append(np.mean(scores_realistic))
            else:
                # attack
                fp_dataset_attack = fp_dataset.drop(attr, axis=1)
                data_original_preprocessed = data.drop(attr, axis=1)
                fp_dataset_attack = pd.get_dummies(fp_dataset_attack)
                data_original_preprocessed = pd.get_dummies(data_original_preprocessed)
                if 'breast_left' in fp_dataset_attack:
                    fp_dataset_attack = fp_dataset_attack.drop(['breast_left'], axis=1)
                if 'irradiat_yes' in fp_dataset_attack:
                    fp_dataset_attack = fp_dataset_attack.drop(['irradiat_yes'], axis=1)
                if 'breast_left' in data_original_preprocessed:
                    data_original_preprocessed = data_original_preprocessed.drop(['breast_left'], axis=1)
                if 'irradiat_yes' in data_original_preprocessed:
                    data_original_preprocessed = data_original_preprocessed.drop(['irradiat_yes'], axis=1)

                if len(data_original_preprocessed.columns.difference(fp_dataset_attack.columns)) != 0:
                    data_original_preprocessed = data_original_preprocessed.drop(data_original_preprocessed.columns.difference(
                        fp_dataset_attack.columns), axis=1)
                if len(fp_dataset_attack.columns.difference(data_original_preprocessed.columns)) != 0:
                    fp_dataset_attack = fp_dataset_attack.drop(fp_dataset_attack.columns.difference(
                        data_original_preprocessed.columns), axis=1)

                model2 = DecisionTreeClassifier(random_state=random_state, criterion='entropy', max_depth=2)
                #scores = cross_val_score(model2, fp_dataset_attack.values, target, cv=10)
                # todo fingerprinted scores
                #scores_realistic = fingerprint_cross_val_score(model2, data_original_preprocessed.values,
                #                                               fp_dataset_attack.values, target, cv=10)
                if attr not in score:
                    score[attr] = []
                #score[attr].append(np.mean(scores))
                if attr not in score_realistic:
                    score_realistic[attr] = []
                #score_realistic[attr].append(np.mean(scores_realistic))

            secret_key = secret_key - 3
    print("Fingerprinted full: " + str(accuracy_fingerprinted_full))
    pprint(score)
    print("Averages")
    pprint({i: np.mean(score[i]) for i in score})
    pprint({i: np.mean(score_realistic[i]) for i in score_realistic})


def predicting_with_one_attr():
    data = pd.read_csv('datasets/breast_cancer_full.csv', index_col='Id')
    target = data.values[:, (len(data.columns) - 1)]
    data = data.drop("recurrence", axis=1)

    # retrieving performance from the previous runs
    accuracy_original = 0.7168  # for decision tree: max_depth = 2, criterion = 'entropy'
    gamma = 5
    n_removed_attr = 7

    # try removing each attribute and record the results
    score = dict()
    score_realistic = dict()
    n_exp = 1
    random_state = 25
    attributes = list(itertools.combinations(data.columns, n_removed_attr))
    attributes = [list(combo) for combo in attributes]
    for attr_combination in attributes:
        # calculate utility of attacked fingerprinted data set
        secret_key = 3255  # increase every run
        for n in range(n_exp):
            # fingerprint the data
            scheme = CategoricalNeighbourhood(gamma=gamma, xi=2, fingerprint_bit_length=8)
            fp_dataset = scheme.insertion(dataset_name="breast_cancer", buyer_id=1, secret_key=secret_key)
            fp_dataset = fp_dataset.drop("Id", axis=1)

            # attack
            fp_dataset_attack = fp_dataset.drop(attr_combination, axis=1)
            # data_original_preprocessed = data.drop(attr_combination, axis=1)
            fp_dataset_attack = pd.get_dummies(fp_dataset_attack)
            # data_original_preprocessed = pd.get_dummies(data_original_preprocessed)
            # todo: remove
            data_original_preprocessed = pd.get_dummies(data)
            if 'breast_left' in fp_dataset_attack:
                fp_dataset_attack = fp_dataset_attack.drop(['breast_left'], axis=1)
            if 'irradiat_yes' in fp_dataset_attack:
                fp_dataset_attack = fp_dataset_attack.drop(['irradiat_yes'], axis=1)

            if len(data_original_preprocessed.columns.difference(fp_dataset_attack.columns)) != 0:
                data_original_preprocessed = data_original_preprocessed.drop(
                    data_original_preprocessed.columns.difference(
                        fp_dataset_attack.columns), axis=1)
            if len(fp_dataset_attack.columns.difference(data_original_preprocessed.columns)) != 0:
                fp_dataset_attack = fp_dataset_attack.drop(fp_dataset_attack.columns.difference(
                    data_original_preprocessed.columns), axis=1)

            model2 = DecisionTreeClassifier(random_state=random_state, criterion='entropy', max_depth=2)
            print(fp_dataset_attack.columns)
            scores = cross_val_score(model2, fp_dataset_attack.values, target, cv=10)
            scores_realistic = fingerprint_cross_val_score(model2, data_original_preprocessed.values,
                                                           fp_dataset_attack.values, target, cv=10)
            if str(attr_combination) not in score:
                score[str(attr_combination)] = []
            score[str(attr_combination)].append(np.mean(scores))
            if str(attr_combination) not in score_realistic:
                score_realistic[str(attr_combination)] = []
            score_realistic[str(attr_combination)].append(np.mean(scores_realistic))

            secret_key = secret_key - 3
    pprint(score)
    pprint(score_realistic)


if __name__ == '__main__':
    demo_exp()
    predicting_with_one_attr()
