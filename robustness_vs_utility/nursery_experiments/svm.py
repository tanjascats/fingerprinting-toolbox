# import data
# choose a classifier
# calculate (or retrieve) the utility of the original data set
# fingerprint -> choose 1 gamma
# calculate (or retrieve) the average utility of the fingerprinted data set (full)
# remove 2 columns; calculate the average utility
#   -> repeat n times
#        -> repeat for each pair of attributes (or a subset if there is too much of them)
# plot how the utility changes by removing which attribute

import sys
sys.path.append("/home/sarcevic/fingerprinting-toolbox/")
import numpy as np
from attacks.bit_flipping_attack import BitFlippingAttack
import pandas as pd
from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood
# apply normalization if negative values are not allowed
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from pprint import pprint
import itertools
from datetime import datetime
import random


from sdv import SDV
from sdv import Metadata

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


def hyperparameter_tuning():
    data = pd.read_csv('datasets/nursery_full.csv', index_col='Id')
    target = data.values[:, (len(data.columns) - 1)]
    data = data.drop("target", axis=1)
    # one-hot encode
    data_dumm = pd.get_dummies(data)
    # a bit more preprocessing
    data_dumm = data_dumm.drop(['finance_inconv'], axis=1)
    # standard_scaler = StandardScaler()
    # data_scaled = standard_scaler.fit_transform(data_dumm)

    kernel_range = ['linear', 'poly', 'rbf', 'sigmoid']
    param_dist = dict(kernel=kernel_range)
    # hyperparameter random search
    # take the best accuracy from 10-fold cross validation as a benchmark performance
    model = SVC()
    rand_search = GridSearchCV(model, param_dist, cv=10, scoring="accuracy")
    rand_search.fit(data_dumm, target)
    best_params = rand_search.best_params_
    print(best_params)
    print(rand_search.best_score_)
    print(rand_search.best_estimator_)
    return rand_search.best_estimator_


def demo_exp(gamma, n_removed_attr):  # [20, 10, 7, 3]
    data = pd.read_csv('datasets/nursery_full.csv', index_col='Id')
    target = data.values[:, (len(data.columns) - 1)]
    data = data.drop("target", axis=1)

    # one-hot encode
    data_dumm = pd.get_dummies(data)
    # a bit more preprocessing
    data_dumm = data_dumm.drop(['finance_inconv'], axis=1)
    standard_scaler = StandardScaler()
    # data_scaled = standard_scaler.fit_transform(data_dumm)

    # retrieving performance from the previous runs
    accuracy_original = 0.8882716049382716  # {'kernel': 'rbf'}
    # accuracy_fingerprinted_full = 0.6994

    # try removing each attribute and record the results
    score = dict()
    score_realistic = dict()
    # todo: randomly choose a combination of attributes to remove (or reduce number of experiments)
    n_exp = 25
    random_state = 25
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
            scheme = CategoricalNeighbourhood(gamma=gamma, xi=2, fingerprint_bit_length=32)
            fp_dataset = scheme.insertion(dataset_name="nursery", buyer_id=1, secret_key=secret_key)
            fp_dataset = fp_dataset.drop("Id", axis=1)

            if attr_combination[0] is 'full':
                fp_dataset_dumm = pd.get_dummies(fp_dataset)
                fp_dataset_dumm = fp_dataset_dumm.drop(['finance_inconv'], axis=1)
                data_dumm_preprocessed = data_dumm

                if len(data_dumm_preprocessed.columns.difference(fp_dataset_dumm.columns)) != 0:
                    data_dumm_preprocessed = data_dumm_preprocessed.drop(data_dumm_preprocessed.columns.difference(
                        fp_dataset_dumm.columns), axis=1)
                if len(fp_dataset_dumm.columns.difference(data_dumm_preprocessed.columns)) != 0:
                    fp_dataset_dumm = fp_dataset_dumm.drop(fp_dataset_dumm.columns.difference(
                        data_dumm_preprocessed.columns), axis=1)
                # scale
                fp_dataset_dumm = standard_scaler.fit_transform(fp_dataset_dumm)

                model = SVC(random_state=random_state, kernel='rbf')
                # score on fingerprinted test data
                scores = cross_val_score(model, fp_dataset_dumm, target, cv=10)
                # todo: score on original data
                scores_realistic = fingerprint_cross_val_score(model, data_dumm_preprocessed.values, fp_dataset_dumm, target, cv=10)
                if 'full' not in score:
                    score['full'] = []
                score['full'].append(np.mean(scores))
                if 'full' not in score_realistic:
                    score_realistic['full'] = []
                score_realistic['full'].append(np.mean(scores_realistic))

            else:
                # attack
                fp_dataset_attack = fp_dataset.drop(attr_combination, axis=1)
                data_original_preprocessed = data.drop(attr_combination, axis=1)
                fp_dataset_attack = pd.get_dummies(fp_dataset_attack)
                data_original_preprocessed = pd.get_dummies(data_original_preprocessed)
                if 'finance_inconv' in fp_dataset_attack:
                    fp_dataset_attack = fp_dataset_attack.drop(['finance_inconv'], axis=1)
                if 'finance_inconv' in data_original_preprocessed:
                    data_original_preprocessed = data_original_preprocessed.drop(['finance_inconv'], axis=1)

                if len(data_original_preprocessed.columns.difference(fp_dataset_attack.columns)) != 0:
                    data_original_preprocessed = data_original_preprocessed.drop(
                        data_original_preprocessed.columns.difference(
                            fp_dataset_attack.columns), axis=1)
                if len(fp_dataset_attack.columns.difference(data_original_preprocessed.columns)) != 0:
                    fp_dataset_attack = fp_dataset_attack.drop(fp_dataset_attack.columns.difference(
                        data_original_preprocessed.columns), axis=1)
                # scale data
                fp_dataset_attack = standard_scaler.fit_transform(fp_dataset_attack)
                data_original_preprocessed = standard_scaler.fit_transform(data_original_preprocessed)

                model2 = SVC(random_state=random_state, kernel='rbf')
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
    f = open("robustness_vs_utility/nursery_experiments/log.txt", "a+")
    pprint(score, f)
    pprint("Averages", f)
    pprint({i: np.mean(score[i]) for i in score}, f)
    pprint("Timestamp: " + str(datetime.today()), f)
    pprint("------------------------------------------\n------------------------------------------", f)
    f.close()

    with open("robustness_vs_utility/nursery_experiments/log_refined.txt", "a+") as log_refined:
        log_refined.write("\nRemoving " + str(n_removed_attr) + " attributes\n")
        log_refined.write("SVM\n")
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

    with open("robustness_vs_utility/nursery_experiments/log_formatted", "a+") as log_formatted:
        # experiment_results_data['decision tree'][0][3] = {
        gamma_formatted = {3: 0, 7: 1, 10: 2, 20: 3}
        log_formatted.write("experiment_results_data['svm'][" + str(gamma_formatted[gamma]) +
                            "][" + str(n_removed_attr) + "] = {")
        keys = map(str, {i: np.mean(score_realistic[i]) for i in score_realistic})
        items = (key + ": " + str(np.mean(score_realistic[key])) for key in keys)
        line = ",\n".join(items)
        line = line.replace("[", "(")
        line = line.replace("]", ")")
        line += "}\n"
        log_formatted.write(line)


def feature_selection(gamma):
    data = pd.read_csv('datasets/nursery_full.csv', index_col='Id')
    target = data.values[:, (len(data.columns) - 1)]
    data = data.drop("target", axis=1)
    label_encoder = LabelEncoder()
    for col in data.columns:
        data[col] = label_encoder.fit_transform(data[col])
    baseline = pd.value_counts(target)
    print("Baseline: " + str(baseline))

    # add a loop for number of experiments with differently fingerprinted data
    # fingerprinted
    secret_key = 25
    n_exp = 100
    scores = {i: [] for i in range(1, len(data.columns))}
    for e in range(n_exp):
        scheme = CategoricalNeighbourhood(gamma=gamma, xi=2, fingerprint_bit_length=32)
        fp_dataset = scheme.insertion(dataset_name="nursery", buyer_id=1, secret_key=secret_key)
        fp_dataset = fp_dataset.drop("Id", axis=1)

        label_encoder = LabelEncoder()
        for col in fp_dataset.columns:
            fp_dataset[col] = label_encoder.fit_transform(fp_dataset[col])
        for n_features in range(1, len(fp_dataset.columns)):
            feature_model = SelectKBest(chi2, k=n_features)
            features = feature_model.fit_transform(fp_dataset, target)
            print(fp_dataset.columns.values[feature_model.get_support()])
            #print(feature_model.get_support())
            #print(np.invert(feature_model.get_support()))
            #exit()

            model = SVC(random_state=0, kernel='rbf')
            s = cross_val_score(model, features, target, cv=10)
            #print(np.mean(s))
            scores[n_features].append(np.mean(s))
        secret_key += 2

    return scores


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
    n_exp = 7
    # LOAD DATA
    data = pd.read_csv('datasets/nursery_full.csv', index_col='Id')
    target = data['target']

    scores = []
    for e in range(n_exp):
        # FINGERPRINT
        secret_key = 1
        scheme = CategoricalNeighbourhood(gamma=gamma, xi=2, fingerprint_bit_length=16)
        fp_dataset = scheme.insertion(dataset_name="nursery", buyer_id=0, secret_key=secret_key)
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
        model = SVC(random_state=0, kernel='rbf')
        score = cross_val_score(model, fp_dataset_attacked.drop('target', axis=1), fp_dataset_attacked['target'], cv=10)
        scores.append(np.mean(score))
        results.loc[len(results)] = {'#removed': percentage_removed, 'accuracy': np.mean(score), 'gamma': gamma,
                                     'classifier': 'svm'}
    print(np.mean(scores))
    return results.copy()
#     data = pd.DataFrame(columns=['#removed', 'removed_attr', 'accuracy', 'gamma', 'classifier'])


def bit_flipping_attack(gamma, percentage_flip, results):
    n_exp = 10
    # LOAD DATA
    data = pd.read_csv('datasets/nursery_full.csv', index_col='Id')
    data = preprocess(data)
    target = data['target']

    # fingerprint the data
    score = []
    for e in range(n_exp):
        secret_key = 0
        scheme = CategoricalNeighbourhood(gamma=gamma, xi=2, fingerprint_bit_length=32)
        fp_dataset = scheme.insertion('nursery', buyer_id=0, secret_key=secret_key)
        secret_key += 3

        # attack
        attack = BitFlippingAttack()
        fp_dataset_attack = attack.run(fp_dataset, percentage_flip).drop('Id', axis=1)
        fp_dataset_attack = preprocess(fp_dataset_attack)

        model = SVC(random_state=0, kernel='rbf')
        scores = cross_val_score(model, fp_dataset_attack, target, cv=10)
        score.append(np.mean(scores))
        results.loc[len(results)] = {'#removed': percentage_flip, 'accuracy': np.mean(scores), 'gamma': gamma,
                                     'classifier': 'svm'}

    return results.copy()


def transferable_feature_selection(gamma):
    # performing a feature selection for one target
    # choosing a different target and calculating the performance with the given features
    data = pd.read_csv('datasets/nursery_full.csv', index_col='Id')
    target = data.values[:, (len(data.columns) - 1)]
    data = data.drop("target", axis=1)
    label_encoder = LabelEncoder()
    for col in data.columns:
        data[col] = label_encoder.fit_transform(data[col])
    baseline = pd.value_counts(target)
    print("Baseline: " + str(baseline))

    secret_key = 25
    n_exp = 5
    scores_on_different_target = {i: [] for i in range(1, len(data.columns))}
    for e in range(n_exp):
        scheme = CategoricalNeighbourhood(gamma=gamma, xi=2, fingerprint_bit_length=32)
        fp_dataset = scheme.insertion(dataset_name="nursery", buyer_id=1, secret_key=secret_key)
        fp_dataset = fp_dataset.drop("Id", axis=1)

        label_encoder = LabelEncoder()
        for col in fp_dataset.columns:
            fp_dataset[col] = label_encoder.fit_transform(fp_dataset[col])
        for n_features in range(1, len(fp_dataset.columns)):
            feature_model = SelectKBest(chi2, k=n_features)
            feature_model.fit_transform(fp_dataset, target)
            features_names = fp_dataset.columns.values[feature_model.get_support()]
            print(features_names)
            # print(feature_model.get_support())
            # print(np.invert(feature_model.get_support()))
            # exit()

            # now adapt the model to the new target
            new_model = SVC(kernel='rbf')
            # define new target
            new_target_name = 'parents'
            new_target = fp_dataset[new_target_name]
            # and exchange the features: features - new target + old target
            new_features_names = features_names[features_names != new_target_name]
            new_features = fp_dataset[new_features_names]
            new_features['target'] = label_encoder.fit_transform(target)

            s = cross_val_score(new_model, new_features, new_target, cv=10)
            scores_on_different_target[n_features].append(np.mean(s))

        secret_key += 2
    return scores_on_different_target.copy()


def synth(data, amount):
    input_df = data
    tables = {'ftable': input_df}

    metadata = Metadata()
    metadata.add_table('ftable', data=tables['ftable'])

    sdv = SDV()
    sdv.fit(metadata, tables)

    samples = sdv.sample_all(amount)  # here: number of synthetic samples
    df = samples['ftable']
    df = df[input_df.columns]
    return df


def horizontal_with_imputation(gamma, percentage_removed, results):
    n_exp = 3
    # LOAD DATA
    data = pd.read_csv('datasets/nursery_full.csv', index_col='Id')
    target = data['target']

    scores = []
    for e in range(n_exp):
        # FINGERPRINT
        secret_key = 1
        scheme = CategoricalNeighbourhood(gamma=gamma, xi=2, fingerprint_bit_length=16)
        fp_dataset = scheme.insertion(dataset_name="nursery", buyer_id=0, secret_key=secret_key)
        fp_dataset = fp_dataset.drop("Id", axis=1)
        secret_key += 3

        fp_dataset_full = fp_dataset.copy()
        fp_dataset_full['target'] = target

        # ATTACK
        fp_dataset_attacked = fp_dataset_full.copy()
        random_sample = sorted(random.sample([i for i in range(len(fp_dataset_full))],
                                             int(percentage_removed * len(fp_dataset_full))))
        for i in random_sample:
            fp_dataset_attacked = fp_dataset_attacked.drop(i, axis=0)
            # todo: imputation
        synthetic_amount = len(fp_dataset) - len(fp_dataset_attacked)
        synthetic_data = synth(fp_dataset_full, synthetic_amount)
        #missing_ids = pd.concat([fp_dataset['Id'], fp_dataset_attacked['Id']]).drop_duplicates(keep=False)
        #synthetic_data = synthetic_data.assign(Id=missing_ids.values)
        imputed_data = fp_dataset_attacked.append(synthetic_data)
        imputed_data = imputed_data.reset_index().drop('index', axis=1)
        # todo preprocess different data
        imputed_data = preprocess(imputed_data)

        # CLASSIFICATION SCORE
        model = SVC(random_state=0, kernel='rbf')
        score = cross_val_score(model, imputed_data.drop('target', axis=1), imputed_data['target'], cv=10)
        scores.append(np.mean(score))
        results.loc[len(results)] = {'#removed': percentage_removed, 'accuracy': np.mean(score), 'gamma': gamma,
                                     'classifier': 'svm'}
    print(np.mean(scores))
    return results.copy()


if __name__ == '__main__':
    #results = pd.DataFrame(columns=['#removed', 'accuracy', 'gamma', 'classifier'])
    gamma = 10
    #for portion in range(0, 100, 5):
    #    portion = portion / 100.0
    #    results = horizontal_with_imputation(gamma, portion, results)
    # save results
    #outfile = open('robustness_vs_utility/nursery_experiments/horizontal_with_imputation_svm_' + str(gamma),
    #               'wb')
    #pickle.dump(results, outfile)
    #outfile.close()
    results = transferable_feature_selection(gamma)
    outfile = open('robustness_vs_utility/nursery_experiments/transferable_feature_selection_svm_parents_10', 'wb')
    pickle.dump(results, outfile)
    outfile.close()
    # best score health 0.7594135802469136
    # best score parents 0.4353395061728395
    # IMPORTANT: the resutls ire in exchanged files