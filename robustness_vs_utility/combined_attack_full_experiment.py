# FULL EXPERIMENT FOR COMBINED ATTACK
# - ROBUSTNESS
# - UTILITY: decision tree, svm, gradient boosting, logistic regression, knn
import sys
sys.path.append("/home/sarcevic/fingerprinting-toolbox/")

from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood
from attacks.combined_attack import CombinedAttack

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from pprint import pprint
from time import time
import string

# # ------------------------------------------- # #
# # --------------- METHODOLOGY --------------- # #
# # ------------------------------------------- # #
# (1) fingerprint insertion
# (1.1) sanity check (detection must give a correct suspect)
# (2) perform attack - one of the 12 attack configurations
# (3) detection - record success
# (4) train the classifiers using the preset hyperparameters
# (4.1) decision tree
# (4.2) svm
# (4.3) gradient boosting
# (4.4) logistic regression
# (4.5) knn
# (5) record the results
# (6) plot the results

# # ------------------------------------------- # #
# # ------------- CONFIGURATION --------------- # #
# # ------------------------------------------- # #
hyperparam_config = {'breast_cancer_decision_tree': {'criterion': 'entropy', 'max_depth': 2},
                     'breast_cancer_svm': {'kernel': 'poly'},
                     'breast_cancer_logistic_regression': {'solver': 'saga', 'C': 90},
                     'breast_cancer_knn': {'n_neighbors': 19, 'algorithm': 'kd_tree'},
                     'breast_cancer_gradient_boosting': {'n_estimators': 200, 'loss': 'exponential', 'criterion': 'mae'},
                     'german_credit_decision_tree': {'criterion': 'entropy', 'max_depth': 9},
                     'german_credit_gradient_boosting': {'n_estimators': 130, 'loss': 'deviance', 'criterion': 'friedman_mse'},
                     'german_credit_knn': {'n_neighbors': 21, 'algorithm': 'kd_tree'},
                     'german_credit_logistic_regression': {'C': 10, 'solver': 'liblinear'},
                     'german_credit_svm': {'kernel': 'rbf'},
                     'nursery_decision_tree': {'criterion': 'entropy', 'max_depth': 22},
                     'nursery_gradient_boosting': {'criterion': 'friedman_mse', 'n_estimators': 100, 'loss': 'deviance'},
                     'nursery_knn': {'n_neighbors': 8, 'algorithm': 'auto'},
                     'nursery_logistic_regression': {'C': 20, 'solver': 'newton-cg'},
                     'nursery_svm': {'kernel': 'rbf'}}
attack_configurations = [[0.9, 0, 0.1], [0.7, 0, 0.1], [0.9, 0, 0.3], [0.7, 0, 0.3],
                         [0.9, 1, 0.1], [0.7, 1, 0.1], [0.9, 1, 0.3], [0.7, 1, 0.3],
                         [0.9, 3, 0.1], [0.7, 3, 0.1], [0.9, 3, 0.3], [0.7, 3, 0.3]]
fp_bit_length = {'breast_cancer': 8, 'german_credit': 8, 'nursery': 16}
gammae = {'breast_cancer': [1, 2, 3, 5], 'german_credit': [3, 7, 10, 20], 'nursery': [5, 10, 20, 30]}
target_name = {'breast_cancer': 'recurrence', 'german_credit': 'target', 'nursery': 'target'}
classifiers = ['decision_tree', 'svm', 'gradient_boosting', 'logistic_regression', 'knn']

# # ------------------------------------------- # #
# # --------------- REPETITIONS --------------- # #
# # ------------------------------------------- # #
n_fp_experiments = 8  # 15 #25  # number of times we run fp insertion (and classification)
n_experiments = 10  # 6 #10  # number of times we attack the same fingerprinted file

STATUS = ''


def preprocess(data):
    label_encoder = LabelEncoder()
    for col in data.columns:
        data[col] = label_encoder.fit_transform(data[col])
    if 'Id' in data.columns:
        data = data.drop(['Id'], axis=1)
    return data


def run(attack_config, data, full_target, gamma):
    print(data)
    count_detected = 0
    score_classification = {'gradient_boosting': [], 'svm': [], 'decision_tree': [], 'knn': [],
                            'logistic_regression': []}
    secret_key = 11

    # (1) fingerprint insertion
    scheme = CategoricalNeighbourhood(gamma=gamma, xi=1, fingerprint_bit_length=fp_bit_length[data])
    for i in range(n_fp_experiments):
        fingerprinted = scheme.insertion(dataset_name=data, buyer_id=0, secret_key=secret_key+i)

        # (1.1) sanity check (detection must give a correct suspect)
        suspect = scheme.detection(dataset_name=data, real_buyer_id=0, secret_key=secret_key+i, dataset=fingerprinted)
        if suspect != 0:
            print('Warning! Sanity check fail: the suspect not detected from the originally fingerprinted data.')
            exit()

        # (2) perform attack - one of the 12 attack configurations
        attack = CombinedAttack()
        for j in range(n_experiments):
            attacked = attack.run(dataset=fingerprinted, fraction_subset=attack_config[0],
                                  number_of_columns=attack_config[1], fraction_flipping=attack_config[2],
                                  random_seed=gamma*(secret_key+i+j)) # remaining potion, deleted absolute, flipped portion
            # (3) detection
            suspect = scheme.detection(dataset_name=data, real_buyer_id=0, secret_key=111, dataset=attacked)
            if suspect == 0:
                print("CORRECT!")
                count_detected += 1
            else:
                print("FALSE!")
            print("STATUS: " + STATUS)
        # (4) train the classifiers using the preset hyperparameters
        attacked = preprocess(attacked)

        # (4.1) decision tree
        hyperparam = hyperparam_config[data + '_decision_tree']
        model = DecisionTreeClassifier(random_state=0, criterion=hyperparam['criterion'],
                                       max_depth=hyperparam['max_depth'])
        target = full_target[attacked.index]
        scores = cross_val_score(model, attacked, target, cv=10)
        score_classification['decision_tree'].append(np.mean(scores))

        # (4.2) svm
        hyperparam = hyperparam_config[data + '_svm']
        model = SVC(random_state=0, kernel=hyperparam['kernel'])
        scores = cross_val_score(model, attacked, target, cv=10)
        score_classification['svm'].append(np.mean(scores))

        # (4.3) gradient boosting
        hyperparam = hyperparam_config[data + '_gradient_boosting']
        model = GradientBoostingClassifier(random_state=0, n_estimators=hyperparam['n_estimators'],
                                           loss=hyperparam['loss'], criterion=hyperparam['criterion'])
        scores = cross_val_score(model, attacked, target, cv=10)
        score_classification['gradient_boosting'].append(np.mean(scores))

        # (4.4) logistic regression
        hyperparam = hyperparam_config[data + '_logistic_regression']
        model = LogisticRegression(random_state=0, solver=hyperparam['solver'], C=hyperparam['C'])
        scores = cross_val_score(model, attacked, target, cv=10)
        score_classification['logistic_regression'].append(np.mean(scores))

        # (4.5) knn
        hyperparam = hyperparam_config[data + "_knn"]
        model = KNeighborsClassifier(n_neighbors=hyperparam['n_neighbors'], algorithm=hyperparam['algorithm'])
        for col in attacked.columns:
            print(attacked[col].value_counts())
        print(target)
        scores = cross_val_score(model, attacked, target, cv=10)
        score_classification['knn'].append(np.mean(scores))

    score_robustness = count_detected / (n_fp_experiments*n_experiments)
    return score_robustness, score_classification


def plot(robustness_data, utility_data, data_name):
    sns.set_style('whitegrid')
    sns.set_palette('cubehelix')  # icefire, Spectral, cubehelix
    robustness_data.plot(kind='bar', stacked=False, width=0.75, ylim=(0, 1))
    plt.subplots_adjust(bottom=0.35)
    # plt.title('')
    plt.xlabel('attack model')
    xtick_labels = ["baseline",
                    "(0%; 10%; 10%)", '(0%; 30%; 10%)', '(0%; 10%; 30%)', '(0%; 30%; 30%)',
                    "(10%; 10%; 10%)", '(10%; 30%; 10%)', '(10%; 10%; 30%)', '(10%; 30%; 30%)',
                    "(30%; 10%; 10%)", '(30%; 30%; 10%)', '(30%; 10%; 30%)', '(30%; 30%; 30%)']
    plt.xticks([i for i in range(len(robustness_data))], xtick_labels, rotation=90)
    if data_name == 'breast_cancer':
        plt.ylabel('false miss')

    plt.show()

    sns.set_palette('colorblind')
    for gamma_idx, utility_result in enumerate(utility_data):
        if data_name == 'breast_cancer' and gamma_idx == 0:
            utility_result.plot(kind='line', marker='o', linewidth=0.2)
            plt.subplots_adjust(bottom=0.5)
        else:
            utility_result.plot(kind='line', marker='o', linewidth=0.2, legend=False)
            plt.subplots_adjust(bottom=0.5)
        plt.xticks([i for i in range(len(robustness_data))], ['' for i in range(len(robustness_data))])
        if gamma_idx == 0:
            title = ' '.join(list(map(str.capitalize, data_name.split('_'))))
            plt.title(title)
        if data_name == 'breast_cancer':
            plt.ylabel('gamma=' + str(gammae[data_name][gamma_idx]) + '\naccuracy')
        else:
            plt.ylabel('gamma=' + str(gammae[data_name][gamma_idx]))
        plt.show()


def config_experiment(data_name):
    start = time()
    results_robustness = dict()

    results_robustness[data_name] = pd.DataFrame(columns=gammae[data_name], index=list(map(str, attack_configurations)))
    data = pd.read_csv('datasets/' + data_name + '_full.csv', index_col='Id')
    data = preprocess(data)
    full_target = data[target_name[data_name]]

    for i, gamma in enumerate(gammae[data_name]):
        results_classification = pd.DataFrame(index=list(map(str, attack_configurations)), columns=classifiers)
        print("##############\nGAMMA=" + str(gamma) + "\n##############")
        for j, attack_config in enumerate(attack_configurations):
            iteration_start = time()
            score_robustness, score_classification = run(data=data_name, attack_config=attack_config,
                                                         full_target=full_target, gamma=gamma)
            results_robustness[data_name][gamma][str(attack_config)] = score_robustness
            for classif in score_classification:
                results_classification[classif][str(attack_config)] = np.mean(score_classification[classif])
            STATUS = str(i * len(attack_configurations) + j + 1) + "/" + \
                     str(len(gammae[data_name]) * len(attack_configurations)) + "\n- time of the last iteration: " + \
                     str(int(time() - iteration_start)) + " seconds\n- time passed from the beginning: " + \
                     str(int(time() - start)) + " seconds."
            print("STATUS: " + STATUS)
        results_classification.to_csv("robustness_vs_utility/results/classification_" + data_name + "_" + str(gamma) +
                                      "_" + str(int(time())) + ".csv")
        # results_robustness[data_name].to_csv("robustness_vs_utility/results/robustness_intermediate_" + data_name + "_"
        #                                     + str(int(time())) + ".csv")
    pprint(results_robustness)
    results_robustness[data_name].to_csv("robustness_vs_utility/results/robustness_" + data_name + "_" +
                                         str(int(time())) + ".csv")


if __name__ == '__main__':
    #config_experiment('nursery')
    #exit()
    data_name = 'nursery'
    results_robustness = pd.read_csv("robustness_vs_utility/results/robustness_" + data_name + ".csv",
                                     index_col='Unnamed: 0')
    print(results_robustness)
    for gamma in gammae[data_name]:
        results_robustness[str(gamma)] = results_robustness[str(gamma)].apply(lambda x: 1 - x)
    print(results_robustness)

    results_utility = list()
    for gamma in gammae[data_name]:
        results_utility.append(pd.read_csv("robustness_vs_utility/results/classification_" + str(data_name) + "_" +
                                           str(gamma) + ".csv", index_col='Unnamed: 0'))
    print(results_utility)
    plot(results_robustness, results_utility, data_name)
