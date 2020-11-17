import sys
sys.path.append("/home/sarcevic/fingerprinting-toolbox/")

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood
# apply normalization if negative values are not allowed
from sklearn.preprocessing import StandardScaler
from attacks.bit_flipping_attack import BitFlippingAttack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from pprint import pprint
import itertools
from datetime import datetime
import random


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


'''
Checks the correctness of the workflow of the original experiment

'''
def demo_exp():
    n_removed_attr = 8
    gamma = 2
    n_exp = 2
    random_state = 25

    data = pd.read_csv('datasets/breast_cancer_full.csv', index_col='Id')
    target = data.values[:, (len(data.columns) - 1)]
    data = data.drop("recurrence", axis=1)

    # one-hot encode
    data_dumm = pd.get_dummies(data)
    # a bit more preprocessing
    data_dumm = data_dumm.drop(['breast_left', 'irradiat_yes'], axis=1)

    score = dict()
    score_realistic = dict()
    attributes = list(itertools.combinations(data.columns, n_removed_attr))
    attributes = [list(combo) for combo in attributes]
    if len(attributes) > 2000:
        attributes = random.sample(attributes, int(0.01 * len(attributes)))
    attributes.append(['full'])
    for attr_combination in attributes:
        # calculate utility of attacked fingerprinted data set
        secret_key = 3255  # increase every run
        for n in range(n_exp):
            # fingerprint the data
            scheme = CategoricalNeighbourhood(gamma=gamma, xi=2, fingerprint_bit_length=8)
            fp_dataset = scheme.insertion(dataset_name="breast_cancer", buyer_id=1, secret_key=secret_key)
            fp_dataset = fp_dataset.drop("Id", axis=1)

            if attr_combination[0] is 'full':
                fp_dataset_dumm = pd.get_dummies(fp_dataset)
                fp_dataset_dumm = fp_dataset_dumm.drop(['breast_left', 'irradiat_yes'], axis=1)

                # scale
                standard_scaler = StandardScaler()
                fp_dataset_dumm = standard_scaler.fit_transform(fp_dataset_dumm)

                model = LogisticRegression(random_state=random_state, solver='saga', C=90)
                # score on fingerprinted test data
                scores = cross_val_score(model, fp_dataset_dumm, target, cv=10)
                if 'full' not in score:
                    score['full'] = []
                score['full'].append(np.mean(scores))

            else:
                # attack
                fp_dataset_attack = fp_dataset.drop(attr_combination, axis=1)
                data_original_preprocessed = data.drop(attr_combination, axis=1)
                fp_dataset_attack = pd.get_dummies(fp_dataset_attack)
                data_original_preprocessed = pd.get_dummies(data_original_preprocessed)
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
                # scale data
                standard_scaler = StandardScaler()
                fp_dataset_attack = standard_scaler.fit_transform(fp_dataset_attack)
                data_original_preprocessed = standard_scaler.fit_transform(data_original_preprocessed)

                model2 = LogisticRegression(random_state=random_state, solver='saga', C=90)
                # todo: the following removed for runtime reasons
                # scores = cross_val_score(model2, fp_dataset_attack, target, cv=10)
                scores_realistic = fingerprint_cross_val_score(model2, data_original_preprocessed,
                                                               fp_dataset_attack, target, cv=10)
                # todo: the following removed for runtime reasons
                # if str(attr_combination) not in score:
                #    score[str(attr_combination)] = []
                # score[str(attr_combination)].append(np.mean(scores))
                if str(attr_combination) not in score_realistic:
                    score_realistic[str(attr_combination)] = []
                score_realistic[str(attr_combination)].append(np.mean(scores_realistic))

            secret_key = secret_key - 3
    # print("Fingerprinted full: " + str())
    # score['full'] = [0.7293225779967158]  # for 100 experiments (from demo_exp.py / results in log_refined)
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
        log_refined.write("\nRemoving " + str(n_removed_attr) + " attributes\n")
        log_refined.write("Logistic Regression\n")
        log_refined.write("gamma = " + str(gamma) + "\n")
        log_refined.write("number of experiments = " + str(n_exp) + "\n")
        log_refined.write("Averages:\n")
        keys = map(str, {i: np.mean(score[i]) for i in score})
        items = (key + ": " + str(np.mean(score[key])) for key in keys)
        line = ",\n".join(items)
        log_refined.write(line)

        log_refined.write("\n\nNon-fingerprinted holdout test:\n")
        keys = map(str, {i: np.mean(score_realistic[i]) for i in score_realistic})
        items = (key + ": " + str(np.mean(score_realistic[key])) for key in keys)
        line = ",\n".join(items)
        log_refined.write(line)

        log_refined.write("\nTimestamp: " + str(datetime.today()) + "\n")
        log_refined.write("------------------------------------------\n------------------------------------------")

    with open("robustness_vs_utility/breast_cancer_experiments/log_formatted", "a+") as log_formatted:
        # experiment_results_data['decision tree'][0][3] = {
        gamma_formatted = {1: 0, 2: 1, 3: 2, 5: 3}
        log_formatted.write("experiment_results_data['logistic regression'][" + str(gamma_formatted[gamma]) +
                            "][" + str(n_removed_attr) + "] = {")
        keys = map(str, {i: np.mean(score_realistic[i]) for i in score_realistic})
        items = (key + ": " + str(np.mean(score_realistic[key])) for key in keys)
        line = ",\n".join(items)
        line = line.replace("[", "(")
        line = line.replace("]", ")")
        line += "}\n"
        log_formatted.write(line)

    with open("robustness_vs_utility/breast_cancer_experiments/log_refined.txt", "a+") as log_refined:
        log_refined.write("\nRemoving " + str(n_removed_attr) + " attributes\n")
        log_refined.write("Logistic Regression\n")
        log_refined.write("gamma = " + str(gamma) + "\n")
        log_refined.write("number of experiments = " + str(n_exp) + "\n")
        log_refined.write("Averages:\n")
        keys = map(str, {i: np.mean(score[i]) for i in score})
        items = (key + ": " + str(np.mean(score[key])) for key in keys)
        line = ",\n".join(items)
        log_refined.write(line)

        log_refined.write("\n\nNon-fingerprinted holdout test:\n")
        keys = map(str, {i: np.mean(score_realistic[i]) for i in score_realistic})
        items = (key + ": " + str(np.mean(score_realistic[key])) for key in keys)
        line = ",\n".join(items)
        log_refined.write(line)

        log_refined.write("\nTimestamp: " + str(datetime.today()) + "\n")
        log_refined.write("------------------------------------------\n------------------------------------------")

    with open("robustness_vs_utility/breast_cancer_experiments/log_formatted", "a+") as log_formatted:
        gamma_formatted = {1: 0, 2: 1, 3: 2, 5: 3}
        log_formatted.write("experiment_results_data['logistic regression'][" + str(gamma_formatted[gamma]) +
                            "][" + str(n_removed_attr) + "] = {")
        keys = map(str, {i: np.mean(score_realistic[i]) for i in score_realistic})
        items = (key + ": " + str(np.mean(score_realistic[key])) for key in keys)
        line = ",\n".join(items)
        line = line.replace("[", "(")
        line = line.replace("]", ")")
        line += "}\n"
        log_formatted.write(line)


def full_exp(gamma, n_removed_attr):
    n_exp = 10
    random_state = 25

    data = pd.read_csv('datasets/breast_cancer_full.csv', index_col='Id')
    target = data.values[:, (len(data.columns) - 1)]
    data = data.drop("recurrence", axis=1)

    # one-hot encode
    data_dumm = pd.get_dummies(data)
    # a bit more preprocessing
    data_dumm = data_dumm.drop(['breast_left', 'irradiat_yes'], axis=1)

    score = dict()
    score_realistic = dict()
    attributes = list(itertools.combinations(data.columns, n_removed_attr))
    attributes = [list(combo) for combo in attributes]
    if len(attributes) > 2000:
        attributes = random.sample(attributes, int(0.01*len(attributes)))
    attributes.append(['full'])
    for attr_combination in attributes:
        # calculate utility of attacked fingerprinted data set
        secret_key = 3255  # increase every run
        for n in range(n_exp):
            # fingerprint the data
            scheme = CategoricalNeighbourhood(gamma=gamma, xi=2, fingerprint_bit_length=8)
            fp_dataset = scheme.insertion(dataset_name="breast_cancer", buyer_id=1, secret_key=secret_key)
            fp_dataset = fp_dataset.drop("Id", axis=1)

            if attr_combination[0] is 'full':
                fp_dataset_dumm = pd.get_dummies(fp_dataset)
                fp_dataset_dumm = fp_dataset_dumm.drop(['breast_left', 'irradiat_yes'], axis=1)

                # scale
                standard_scaler = StandardScaler()
                fp_dataset_dumm = standard_scaler.fit_transform(fp_dataset_dumm)

                model = LogisticRegression(random_state=random_state, solver='saga', C=90)
                # score on fingerprinted test data
                scores = cross_val_score(model, fp_dataset_dumm, target, cv=10)
                if 'full' not in score:
                    score['full'] = []
                score['full'].append(np.mean(scores))

            else:
                # attack
                fp_dataset_attack = fp_dataset.drop(attr_combination, axis=1)
                data_original_preprocessed = data.drop(attr_combination, axis=1)
                fp_dataset_attack = pd.get_dummies(fp_dataset_attack)
                data_original_preprocessed = pd.get_dummies(data_original_preprocessed)
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
                # scale data
                standard_scaler = StandardScaler()
                fp_dataset_attack = standard_scaler.fit_transform(fp_dataset_attack)
                data_original_preprocessed = standard_scaler.fit_transform(data_original_preprocessed)

                model2 = LogisticRegression(random_state=random_state, solver='saga', C=90)
                # todo: the following removed for runtime reasons
                # scores = cross_val_score(model2, fp_dataset_attack, target, cv=10)
                scores_realistic = fingerprint_cross_val_score(model2, data_original_preprocessed,
                                                               fp_dataset_attack, target, cv=10)
                # todo: the following removed for runtime reasons
                # if str(attr_combination) not in score:
                #    score[str(attr_combination)] = []
                # score[str(attr_combination)].append(np.mean(scores))
                if str(attr_combination) not in score_realistic:
                    score_realistic[str(attr_combination)] = []
                score_realistic[str(attr_combination)].append(np.mean(scores_realistic))

            secret_key = secret_key - 3
    # print("Fingerprinted full: " + str())
    # score['full'] = [0.7293225779967158]  # for 100 experiments (from demo_exp.py / results in log_refined)
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
        log_refined.write("\nRemoving " + str(n_removed_attr) + " attributes\n")
        log_refined.write("Logistic Regression\n")
        log_refined.write("gamma = " + str(gamma) + "\n")
        log_refined.write("number of experiments = " + str(n_exp) + "\n")
        log_refined.write("Averages:\n")
        keys = map(str, {i: np.mean(score[i]) for i in score})
        items = (key + ": " + str(np.mean(score[key])) for key in keys)
        line = ",\n".join(items)
        log_refined.write(line)

        log_refined.write("\n\nNon-fingerprinted holdout test:\n")
        keys = map(str, {i: np.mean(score_realistic[i]) for i in score_realistic})
        items = (key + ": " + str(np.mean(score_realistic[key])) for key in keys)
        line = ",\n".join(items)
        log_refined.write(line)

        log_refined.write("\nTimestamp: " + str(datetime.today()) + "\n")
        log_refined.write("------------------------------------------\n------------------------------------------")

    with open("robustness_vs_utility/breast_cancer_experiments/log_formatted", "a+") as log_formatted:
        gamma_formatted = {1: 0, 2: 1, 3: 2, 5: 3}
        log_formatted.write("experiment_results_data['logistic regression'][" + str(gamma_formatted[gamma]) +
                            "][" + str(n_removed_attr) + "] = {")
        keys = map(str, {i: np.mean(score_realistic[i]) for i in score_realistic})
        items = (key + ": " + str(np.mean(score_realistic[key])) for key in keys)
        line = ",\n".join(items)
        line = line.replace("[", "(")
        line = line.replace("]", ")")
        line += "}\n"
        log_formatted.write(line)


def original(gamma):
    random_state = 25
    secret_key = 12

    data = pd.read_csv('datasets/breast_cancer_full.csv', index_col='Id')
    target = data.values[:, (len(data.columns) - 1)]

    model = LogisticRegression(random_state=random_state, solver='saga', C=90)

    n_exp = 10
    scores = []
    for e in range(n_exp):
        scheme = CategoricalNeighbourhood(gamma=gamma, xi=2, fingerprint_bit_length=8)
        fp_dataset = scheme.insertion(dataset_name="breast_cancer", buyer_id=1, secret_key=secret_key)
        fp_dataset = fp_dataset.drop("Id", axis=1)

        encoder = LabelEncoder()
        for col in fp_dataset.columns:
            fp_dataset[col] = encoder.fit_transform(fp_dataset[col])
        target = encoder.fit_transform(target)
        s = cross_val_score(model, fp_dataset, target, cv=10)
        scores.append(s)
        secret_key += 3
    print(np.mean(scores))

def preprocess(data):
    # LABEL ENCODER
    label_encoder = LabelEncoder()
    for col in data.columns:
        data[col] = label_encoder.fit_transform(data[col])

    # STANDARD SCALE
    #scaler = StandardScaler()
    #scaled_features = scaler.fit_transform(data)
    #data = pd.DataFrame(scaled_features, index=data.index, columns=data.columns)

    return data


def horizontal_attack(gamma, percentage_removed, results):
    n_exp = 10
    # LOAD DATA
    data = pd.read_csv('datasets/breast_cancer_full.csv', index_col='Id')
    target = data['recurrence']

    scores = []
    for e in range(n_exp):
        # FINGERPRINT
        secret_key = 1
        scheme = CategoricalNeighbourhood(gamma=gamma, xi=2, fingerprint_bit_length=8)
        fp_dataset = scheme.insertion(dataset_name="breast_cancer", buyer_id=0, secret_key=secret_key)
        fp_dataset = fp_dataset.drop("Id", axis=1)
        secret_key += 3

        fp_dataset_full = fp_dataset.copy()
        fp_dataset_full['target'] = target

        # ATTACK
        fp_dataset_attacked = fp_dataset_full.copy()
        random_sample = sorted(random.sample([i for i in range(len(fp_dataset_full))],
                                             int(percentage_removed*len(fp_dataset_full))))
        for i in random_sample:
            fp_dataset_attacked = fp_dataset_attacked.drop(i, axis=0)
        fp_dataset_attacked = preprocess(fp_dataset_attacked)

        # CLASSIFICATION SCORE
        model = LogisticRegression(random_state=1, solver='saga', C=90)
        score = cross_val_score(model, fp_dataset_attacked.drop('target', axis=1), fp_dataset_attacked['target'], cv=10)
        scores.append(np.mean(score))
        results.loc[len(results)] = {'#removed': percentage_removed, 'accuracy': np.mean(score), 'gamma': gamma,
                                     'classifier': 'logistic regression'}
    print(np.mean(scores))
    return results.copy()
#     data = pd.DataFrame(columns=['#removed', 'removed_attr', 'accuracy', 'gamma', 'classifier'])


def bit_flipping_attack(gamma, percentage_flip, results):
    n_exp = 10
    # LOAD DATA
    data = pd.read_csv('datasets/breast_cancer_full.csv', index_col='Id')
    data = preprocess(data)
    target = data['recurrence']

    # fingerprint the data
    score = []
    for e in range(n_exp):
        secret_key = 0
        scheme = CategoricalNeighbourhood(gamma=gamma, xi=2, fingerprint_bit_length=8)
        fp_dataset = scheme.insertion('breast_cancer', buyer_id=0, secret_key=secret_key)
        secret_key += 3

        # attack
        attack = BitFlippingAttack()
        fp_dataset_attack = attack.run(fp_dataset, percentage_flip).drop('Id', axis=1)
        fp_dataset_attack = preprocess(fp_dataset_attack)

        model = LogisticRegression(random_state=0, solver='saga', C=90)
        scores = cross_val_score(model, fp_dataset_attack, target, cv=10)
        score.append(np.mean(scores))
        results.loc[len(results)] = {'#removed': percentage_flip, 'accuracy': np.mean(scores), 'gamma': gamma,
                                     'classifier': 'logistic regression'}

    return results.copy()


if __name__ == '__main__':
    results = pd.DataFrame(columns=['#removed', 'accuracy', 'gamma', 'classifier'])
    gamma = 1
    for portion in range(0, 60, 5):
        portion = portion / 100.0
        results = bit_flipping_attack(gamma, portion, results)
    # save results
    outfile = open('robustness_vs_utility/breast_cancer_experiments/bit-flip_logistic_regression_' + str(gamma), 'wb')
    pickle.dump(results, outfile)
    outfile.close()



# model2 = LogisticRegression(random_state=random_state, solver='saga', C=90)