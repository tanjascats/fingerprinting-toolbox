import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd
import numpy as np


def load_vertical_robustness():
    robustness = [[], [], [], []]
    results = pd.DataFrame(columns=['#removed', 'false miss', 'gamma', 'gamma_id'])
    n_exp = 250
    # !!!! RESULTS FOR BREAST CANCER L=64!!!!
    # gamma = 5
    robustness[0] = n_exp - np.array([250, 250, 250, 250, 243, 232, 226, 201])
    robustness[0] = [i / 250 for i in robustness[0]]
    for r in range(len(robustness[0])):
        results.loc[len(results)] = {'#removed': r, 'false miss': robustness[0][r], 'gamma': 5, 'gamma_id': 0}
    # ----------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------- #
    # gamma = 10
    robustness[1] = n_exp - np.array([250, 250, 247, 230, 190, 163, 127, 87])
    robustness[1] = [i / 250 for i in robustness[1]]
    for r in range(len(robustness[1])):
        results.loc[len(results)] = {'#removed': r, 'false miss': robustness[1][r], 'gamma': 10, 'gamma_id': 1}
    # ----------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------- #
    # gamma = 20
    robustness[2] = n_exp - np.array([250, 231, 185, 111, 63, 34, 26, 7])
    robustness[2] = [i / 250 for i in robustness[2]]
    for r in range(len(robustness[2])):
        results.loc[len(results)] = {'#removed': r, 'false miss': robustness[2][r], 'gamma': 20, 'gamma_id': 2}
    # ----------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------- #
    # gamma = 30
    robustness[3] = n_exp - np.array([250, 167, 73, 28, 17, 16, 6, 4])
    robustness[3] = [i / 250 for i in robustness[3]]
    for r in range(len(robustness[3])):
        results.loc[len(results)] = {'#removed': r, 'false miss': robustness[3][r], 'gamma': 30, 'gamma_id': 3}
    return results


def plot():
    data = pd.DataFrame(columns=['#removed', 'removed_attr', 'accuracy', 'gamma', 'classifier'])
    decision_tree5 = pickle.load(open('robustness_vs_utility/nursery_experiments/decision_tree5', 'rb'))
    for key in decision_tree5:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(decision_tree5[key]),
                               'gamma': 5, 'classifier': 'decision tree'}
    decision_tree10 = pickle.load(open('robustness_vs_utility/nursery_experiments/decision_tree10', 'rb'))
    for key in decision_tree10:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(decision_tree10[key]),
                               'gamma': 10, 'classifier': 'decision tree'}
    decision_tree20 = pickle.load(open('robustness_vs_utility/nursery_experiments/decision_tree20', 'rb'))
    for key in decision_tree20:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(decision_tree20[key]),
                               'gamma': 20, 'classifier': 'decision tree'}
    decision_tree30 = pickle.load(open('robustness_vs_utility/nursery_experiments/decision_tree30', 'rb'))
    for key in decision_tree30:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(decision_tree30[key]),
                               'gamma': 30, 'classifier': 'decision tree'}

    knn5 = pickle.load(open('robustness_vs_utility/nursery_experiments/knn5', 'rb'))
    for key in knn5:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(knn5[key]),
                               'gamma': 5, 'classifier': 'knn'}
    knn10 = pickle.load(open('robustness_vs_utility/nursery_experiments/knn10', 'rb'))
    for key in knn10:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(knn10[key]),
                               'gamma': 10, 'classifier': 'knn'}
    knn20 = pickle.load(open('robustness_vs_utility/nursery_experiments/knn20', 'rb'))
    for key in knn20:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(knn20[key]),
                               'gamma': 20, 'classifier': 'knn'}
    knn30 = pickle.load(open('robustness_vs_utility/nursery_experiments/knn30', 'rb'))
    for key in knn30:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(knn30[key]),
                               'gamma': 30, 'classifier': 'knn'}

    logistic_regression5 = pickle.load(open('robustness_vs_utility/nursery_experiments/logistic_regression5', 'rb'))
    for key in logistic_regression5:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(logistic_regression5[key]),
                               'gamma': 5, 'classifier': 'logistic regression'}
    logistic_regression10 = pickle.load(open('robustness_vs_utility/nursery_experiments/logistic_regression10', 'rb'))
    for key in logistic_regression10:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(logistic_regression10[key]),
                               'gamma': 10, 'classifier': 'logistic regression'}
    logistic_regression20 = pickle.load(open('robustness_vs_utility/nursery_experiments/logistic_regression20', 'rb'))
    for key in logistic_regression20:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(logistic_regression20[key]),
                               'gamma': 20, 'classifier': 'logistic regression'}
    logistic_regression30 = pickle.load(open('robustness_vs_utility/nursery_experiments/logistic_regression30', 'rb'))
    for key in logistic_regression30:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(logistic_regression30[key]),
                               'gamma': 30, 'classifier': 'logistic regression'}

    gradient_boosting5 = pickle.load(open('robustness_vs_utility/nursery_experiments/gradient_boosting5', 'rb'))
    for key in gradient_boosting5:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(gradient_boosting5[key]),
                               'gamma': 5, 'classifier': 'gradient boosting'}
    gradient_boosting10 = pickle.load(open('robustness_vs_utility/nursery_experiments/gradient_boosting10', 'rb'))
    for key in gradient_boosting10:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(gradient_boosting10[key]),
                               'gamma': 10, 'classifier': 'gradient boosting'}
    gradient_boosting20 = pickle.load(open('robustness_vs_utility/nursery_experiments/gradient_boosting20', 'rb'))
    for key in gradient_boosting20:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(gradient_boosting20[key]),
                               'gamma': 20, 'classifier': 'gradient boosting'}
    gradient_boosting30 = pickle.load(open('robustness_vs_utility/nursery_experiments/gradient_boosting30', 'rb'))
    for key in gradient_boosting30:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(gradient_boosting30[key]),
                               'gamma': 30, 'classifier': 'gradient boosting'}

    svm10 = pickle.load(open('robustness_vs_utility/nursery_experiments/svm10', 'rb'))
    for key in svm10:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(svm10[key]),
                               'gamma': 10, 'classifier': 'svm'}
    svm20 = pickle.load(open('robustness_vs_utility/nursery_experiments/svm20', 'rb'))
    for key in svm20:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(svm20[key]),
                               'gamma': 20, 'classifier': 'svm'}
    svm30 = pickle.load(open('robustness_vs_utility/nursery_experiments/svm30', 'rb'))
    for key in svm30:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(svm30[key]),
                               'gamma': 30, 'classifier': 'svm'}

    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.7714, 'gamma': 5,
                           'classifier': 'decision tree'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.7740, 'gamma': 10,
                           'classifier': 'decision tree'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.7716, 'gamma': 20,
                           'classifier': 'decision tree'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.7759, 'gamma': 30,
                           'classifier': 'decision tree'}

    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.8408, 'gamma': 5,
                           'classifier': 'logistic regression'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.8429, 'gamma': 10,
                           'classifier': 'logistic regression'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.8440, 'gamma': 20,
                           'classifier': 'logistic regression'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.8442, 'gamma': 30,
                           'classifier': 'logistic regression'}

    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.7689, 'gamma': 5,
                           'classifier': 'knn'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.7709, 'gamma': 10,
                           'classifier': 'knn'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.7720, 'gamma': 20,
                           'classifier': 'knn'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.7712, 'gamma': 30,
                           'classifier': 'knn'}

    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.9708, 'gamma': 5,
                           'classifier': 'gradient boosting'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.9774, 'gamma': 10,
                           'classifier': 'gradient boosting'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.9805, 'gamma': 20,
                           'classifier': 'gradient boosting'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.9813, 'gamma': 30,
                           'classifier': 'gradient boosting'}

    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.8667, 'gamma': 5,
                           'classifier': 'svm'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.869452, 'gamma': 10,
                           'classifier': 'svm'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.8719, 'gamma': 20,
                           'classifier': 'svm'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.87272, 'gamma': 30,
                           'classifier': 'svm'}

    sns.set(style="whitegrid")
    grid = sns.FacetGrid(data, hue='classifier', col='gamma', height=4)
    grid.map(sns.lineplot, '#removed', 'accuracy', alpha=.7)
    plt.xticks(np.arange(0, 8, step=1))
    plt.legend()

    robustness = load_vertical_robustness()
    robustness['gamma'] = robustness['gamma'].astype('int32')
    grid2 = sns.FacetGrid(robustness, col='gamma', height=2.6)
    grid2.map(sns.lineplot, '#removed', 'false miss', alpha=.7)
    plt.xticks(np.arange(0, 8, step=1))
    plt.show()


def load_horizontal_attack():
    classifiers = ['svm', 'decision_tree', 'gradient_boosting', 'logistic_regression', 'knn']
    gammae = [5, 10, 20, 30]
    results = pd.DataFrame()
    for classifier in classifiers:
        for gamma in gammae:
            try:
                results_intermed = pickle.load(open('robustness_vs_utility/nursery_experiments/'
                                                    'horizontal_' + classifier + "_" + str(gamma), 'rb'))
                if classifier == 'logistic_regression' and gamma == 20:
                    results_intermed['gamma'] = results_intermed['gamma'].replace(10, 20)
            except FileNotFoundError:
                continue
            results = results.append(results_intermed, ignore_index=True)
    # append from all files
    return results


def load_horizontal_with_imputations():
    classifiers = ['svm', 'decision_tree', 'gradient_boosting', 'logistic_regression', 'knn']
    gammae = [5, 10, 20, 30]
    results = pd.DataFrame()
    for classifier in classifiers:
        for gamma in gammae:
            try:
                results_intermed = pickle.load(open('robustness_vs_utility/nursery_experiments/'
                                                    'horizontal_with_imputation_' + classifier + "_" + str(gamma), 'rb'))
                #if classifier == 'logistic_regression' and gamma == 20:
                #    results_intermed['gamma'] = results_intermed['gamma'].replace(10, 20)
            except FileNotFoundError:
                print('FILE NOT FOUND')
                continue
            results = results.append(results_intermed, ignore_index=True)
    # append from all files
    return results


def load_horizontal_robustness():
    n_exp = 250
    robustness = [[], [], [], []]
    results = pd.DataFrame(columns=['#removed', 'false miss', 'gamma', 'gamma_id'])
    # !!!! RESULTS FOR L=64 !!!!
    # gamma = 5
    robustness[0] = n_exp - np.flip(np.array([0, 0, 83, 221, 245, 248, 250, 250, 250, 250, 250, 250, 250, 250, 250,
                                              250, 250, 250, 250, 250]))
    robustness[0] = [i / n_exp for i in robustness[0]]
    for r in range(len(robustness[0])):
        results.loc[len(results)] = {'#removed': r * 0.05, 'false miss': robustness[0][r], 'gamma': 5, 'gamma_id': 0}
    # ----------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------- #
    # gamma = 10
    robustness[1] = n_exp - np.flip(np.array([0, 0, 0, 8, 74, 165, 213, 237, 247, 247, 250, 249, 250, 250, 250, 250, 250, 250, 250, 250]))
    robustness[1] = [i / n_exp for i in robustness[1]]
    for r in range(len(robustness[1])):
        results.loc[len(results)] = {'#removed': r * 0.05, 'false miss': robustness[1][r], 'gamma': 10, 'gamma_id': 0}
    # ----------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------- #
    # gamma = 20
    robustness[2] = n_exp - np.flip(np.array([0, 0, 0, 0, 0, 0, 10, 39, 82, 127, 166, 212, 222, 226, 241, 240, 247, 249, 246, 250]))
    robustness[2] = [i / n_exp for i in robustness[2]]
    for r in range(len(robustness[2])):
        results.loc[len(results)] = {'#removed': r * 0.05, 'false miss': robustness[2][r], 'gamma': 20, 'gamma_id': 0}
    # ----------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------- #
    # gamma = 30
    robustness[3] = n_exp - np.flip(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 10, 32, 56, 79, 112, 133, 167, 185, 199, 218, 250]))
    robustness[3] = [i / n_exp for i in robustness[3]]
    for r in range(len(robustness[3])):
        results.loc[len(results)] = {'#removed': r * 0.05, 'false miss': robustness[3][r], 'gamma': 30, 'gamma_id': 0}
    # ----------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------- #
    return results


def plot_horizontal_attack():
    sns.set(style="whitegrid")
    data = load_horizontal_attack()
    data = data[data['#removed'] < 0.95]
    grid = sns.FacetGrid(data, hue='classifier', col='gamma', height=4)
    grid.map(sns.lineplot, '#removed', 'accuracy', alpha=.7)
    plt.xticks(np.arange(0, 1.0, 0.10), np.arange(0, 100, 10))
    plt.legend()

    robustness = load_horizontal_robustness()
    robustness = robustness.rename(columns={'#removed': '%removed'})
    robustness['gamma'] = robustness['gamma'].astype('int32')
    grid2 = sns.FacetGrid(robustness, col='gamma', height=2.6)
    grid2.map(sns.lineplot, '%removed', 'false miss', alpha=.7)
    plt.xticks(np.arange(0, 1.0, 0.10), np.arange(0, 100, 10))
    plt.show()


def load_bit_flipping_attack():
    classifiers = ['svm', 'decision_tree', 'gradient_boosting', 'logistic_regression', 'knn']
    gammae = [5, 10, 20, 30]
    results = pd.DataFrame()
    for classifier in classifiers:
        for gamma in gammae:
            try:
                results_intermed = pickle.load(open('robustness_vs_utility/nursery_experiments/'
                                                    'bit-flip_' + classifier + "_" + str(gamma), 'rb'))
            except FileNotFoundError:
                continue
            results = results.append(results_intermed, ignore_index=True)
    # append from all files
    return results


def load_bit_flip_robustness():
    robustness = [[], [], [], []]
    results = pd.DataFrame(columns=['#removed', 'false miss', 'gamma', 'gamma_id'])
    n_exp = 250

    # !!!! RESULTS FOR BREAST CANCER L=64!!!!
    # --------------------------------------- #
    # gamma = 5
    # --------------------------------------- #
    robustness[0] = n_exp - np.array([250, 250, 250, 250, 250, 250, 250, 248, 250, 241, 230, 210])
    robustness[0] = [i / n_exp for i in robustness[0]]
    for r in range(len(robustness[0])):
        results.loc[len(results)] = {'#removed': r * 0.05, 'false miss': robustness[0][r], 'gamma': 5,
                                     'gamma_id': 0}
    # --------------------------------------- #
    # gamma = 10
    # --------------------------------------- #
    robustness[1] = n_exp - np.array([250, 250, 250, 248, 247, 242, 233, 205, 177, 136, 101, 60])
    robustness[1] = [i / n_exp for i in robustness[1]]
    for r in range(len(robustness[1])):
        results.loc[len(results)] = {'#removed': r * 0.05, 'false miss': robustness[1][r], 'gamma': 10,
                                     'gamma_id': 1}
    # --------------------------------------- #
    # gamma = 20
    # --------------------------------------- #
    robustness[2] = n_exp - np.array([250, 248, 232, 203, 146, 130, 71, 42, 18, 6, 1, 0])
    robustness[2] = [i / n_exp for i in robustness[2]]
    for r in range(len(robustness[2])):
        results.loc[len(results)] = {'#removed': r * 0.05, 'false miss': robustness[2][r], 'gamma': 20,
                                     'gamma_id': 2}
    # --------------------------------------- #
    # gamma = 30
    # --------------------------------------- #
    robustness[3] = n_exp - np.array([250, 198, 118, 68, 46, 24, 7, 3, 2, 0, 0, 0])
    robustness[3] = [i / n_exp for i in robustness[3]]
    for r in range(len(robustness[2])):
        results.loc[len(results)] = {'#removed': r * 0.05, 'false miss': robustness[3][r], 'gamma': 30,
                                     'gamma_id': 3}
    return results


def plot_bit_flipping_attack():
    sns.set(style="whitegrid")
    data = load_bit_flipping_attack()
    data = data.rename(columns={'#removed': '%flipped'})
    grid = sns.FacetGrid(data, hue='classifier', col='gamma', height=4)
    grid.map(sns.lineplot, '%flipped', 'accuracy', alpha=.7)
    plt.xticks(np.arange(0, 0.6, 0.05), np.arange(0, 60, 5))
    plt.legend()

    robustness = load_bit_flip_robustness()
    robustness = robustness.rename(columns={'#removed': '%flipped'})
    robustness['gamma'] = robustness['gamma'].astype('int32')
    grid2 = sns.FacetGrid(robustness, col='gamma', height=2.6)
    grid2.map(sns.lineplot, '%flipped', 'false miss', alpha=.7)
    plt.xticks(np.arange(0, 0.6, 0.05), np.arange(0, 60, 5))
    plt.xlabel('%removed')
    plt.show()
    plt.show()


def plot_transferable_feature_selection() -> object:
    data = pd.DataFrame(columns=['#removed', 'removed_attr', 'accuracy', 'gamma', 'classifier', 'target'])

    decision_tree10 = pickle.load(open('robustness_vs_utility/nursery_experiments/decision_tree10', 'rb'))
    for key in decision_tree10:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(decision_tree10[key]),
                               'gamma': 10, 'classifier': 'decision tree', 'target': 'original target'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.7740, 'gamma': 10,
                           'classifier': 'decision tree', 'target': 'original target'}

    # target 10
    decision_tree_health = pickle.load(
        open('robustness_vs_utility/nursery_experiments/transferable_feature_selection_score_health_10', 'rb'))
    for key in decision_tree_health:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': '?', 'accuracy': np.mean(decision_tree_health[key]),
                               'gamma': 10, 'classifier': 'decision tree', 'target': "target:'health'"}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': '?', 'accuracy': 0.7456790123, 'gamma': 10,
                           'classifier': 'decision tree', 'target': "target:'health'"}

    # target 2
    decision_tree_parents = pickle.load(
        open('robustness_vs_utility/nursery_experiments/transferable_feature_selection_score_parents_10', 'rb'))
    for key in decision_tree_parents:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': '?', 'accuracy': np.mean(decision_tree_parents[key]),
                               'gamma': 10, 'classifier': 'decision tree', 'target': "target:'parents'"}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': '?', 'accuracy': 0.44243827160493826, 'gamma': 10,
                           'classifier': 'decision tree', 'target': "target:'parents'"}

    knn10 = pickle.load(open('robustness_vs_utility/nursery_experiments/knn10', 'rb'))
    for key in knn10:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(knn10[key]),
                               'gamma': 10, 'classifier': 'knn', 'target': 'original target'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.7709, 'gamma': 10,
                           'classifier': 'knn', 'target': 'original target'}

    knn_parents = pickle.load(open('robustness_vs_utility/nursery_experiments/trans_feature_selection_knn_parents_10',
                                   'rb'))
    for key in knn_parents:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(knn_parents[key]),
                               'gamma': 10, 'classifier': 'knn', 'target': "target:'parents'"}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.4216820987654321, 'gamma': 10,
                           'classifier': 'knn', 'target': "target:'parents'"}

    knn_health = pickle.load(open('robustness_vs_utility/nursery_experiments/trans_feature_selection_knn_health_10',
                                   'rb'))
    for key in knn_health:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(knn_health[key]),
                               'gamma': 10, 'classifier': 'knn', 'target': "target:'health'"}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.7410493827160494, 'gamma': 10,
                           'classifier': 'knn', 'target': "target:'health'"}

    logistic_regression10 = pickle.load(open('robustness_vs_utility/nursery_experiments/logistic_regression10', 'rb'))
    for key in logistic_regression10:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(logistic_regression10[key]),
                               'gamma': 10, 'classifier': 'logistic regression', 'target': 'original target'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.8429, 'gamma': 10,
                           'classifier': 'logistic regression', 'target': 'original target'}

    logistic_regression_parents = pickle.load(
        open('robustness_vs_utility/nursery_experiments/trans_feature_selection_lr_parents_10', 'rb'))
    for key in logistic_regression_parents:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(logistic_regression_parents[key]),
                               'gamma': 10, 'classifier': 'logistic regression', 'target': "target:'parents'"}

    logistic_regression_health = pickle.load(
        open('robustness_vs_utility/nursery_experiments/trans_feature_selection_lr_health_10', 'rb'))
    for key in logistic_regression_health:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(logistic_regression_health[key]),
                               'gamma': 10, 'classifier': 'logistic regression', 'target': "target:'health'"}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.7078703703703704, 'gamma': 10,
                           'classifier': 'logistic regression', 'target': "target:'health'"}

    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.415509, 'gamma': 10,
                           'classifier': 'logistic regression', 'target': "target:'parents'"}

    #gradient_boosting10 = pickle.load(open('robustness_vs_utility/nursery_experiments/gradient_boosting10', 'rb'))
    #for key in gradient_boosting10:
    #    data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(gradient_boosting10[key]),
    #                           'gamma': 10, 'classifier': 'gradient boosting', 'target': 'original target'}
    #data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.9774, 'gamma': 10,
    #                       'classifier': 'gradient boosting', 'target': 'original target'}

    svm10 = pickle.load(open('robustness_vs_utility/nursery_experiments/svm10', 'rb'))
    for key in svm10:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(svm10[key]),
                               'gamma': 10, 'classifier': 'svm', 'target': 'original target'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.869452, 'gamma': 10,
                           'classifier': 'svm', 'target': 'original target'}

    svm_health = pickle.load(
        open('robustness_vs_utility/nursery_experiments/transferable_feature_selection_svm_health_10', 'rb'))
    for key in svm_health:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?",
                               'accuracy': np.mean(svm_health[key]),
                               'gamma': 10, 'classifier': 'svm', 'target': "target:'health'"}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.7594135802469136, 'gamma': 10,
                           'classifier': 'svm', 'target': "target:'health'"}

    svm_parents = pickle.load(
        open('robustness_vs_utility/nursery_experiments/transferable_feature_selection_svm_parents_10', 'rb'))
    for key in svm_parents:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?",
                               'accuracy': np.mean(svm_parents[key]),
                               'gamma': 10, 'classifier': 'svm', 'target': "target:'parents'"}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': "?", 'accuracy': 0.4353395061728395, 'gamma': 10,
                           'classifier': 'svm', 'target': "target:'parents'"}

    sns.set(style="whitegrid")
    grid = sns.FacetGrid(data, hue='target', col='classifier', height=4)
    grid.map(sns.lineplot, '#removed', 'accuracy', alpha=.7)
    plt.xticks(np.arange(0, 8, step=1))
    plt.legend()
    plt.show()


def load_horizontal_with_imputation_robustness():
    n_exp = 120
    robustness = [[], [], [], []]
    results = pd.DataFrame(columns=['#removed', 'false miss', 'gamma', 'gamma_id'])
    # !!!! RESULTS FOR L=64 !!!!
    # gamma = 5
    robustness[0] = n_exp - np.flip(np.array([7, 31, 80, 106, 119, 117, 120, 120, 120, 120, 120, 120, 120, 120, 120,
                                              120, 120, 120, 120, 120]))
    robustness[0] = [i / n_exp for i in robustness[0]]
    for r in range(len(robustness[0])):
        results.loc[len(results)] = {'#removed': r * 0.05, 'false miss': robustness[0][r], 'gamma': 5, 'gamma_id': 0}
    # ----------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------- #
    # gamma = 10
    robustness[1] = n_exp - np.flip(np.array([0, 0, 44, 81, 96, 112, 119, 120, 120, 120, 120, 120, 120, 120, 120, 120,
                                              120, 120, 120, 120]))
    robustness[1] = [i / n_exp for i in robustness[1]]
    for r in range(len(robustness[1])):
        results.loc[len(results)] = {'#removed': r * 0.05, 'false miss': robustness[1][r], 'gamma': 10, 'gamma_id': 1}
    # ----------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------- #
    # gamma = 20
    robustness[2] = n_exp - np.flip(np.array([0, 0, 8, 39, 42, 87, 104, 119, 120, 120, 120, 120, 120, 120, 120, 120,
                                              120, 120, 120, 120]))
    robustness[2] = [i / n_exp for i in robustness[2]]
    for r in range(len(robustness[2])):
        results.loc[len(results)] = {'#removed': r * 0.05, 'false miss': robustness[2][r], 'gamma': 20, 'gamma_id': 2}
    # ----------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------- #
    # gamma = 30
    robustness[3] = n_exp - np.array([120, 120, 120, 120, 120, 120, 120, 120, 120, 117, 98, 97, 74, 66, 44, 18,
                                              9, 1, 0])
    robustness[3] = [i / n_exp for i in robustness[3]]
    for r in range(len(robustness[3])):
        results.loc[len(results)] = {'#removed': r * 0.05, 'false miss': robustness[3][r], 'gamma': 30,
                                     'gamma_id': 3}
    # ----------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------- #
    return results


def plot_horizontal_with_imputation():
    sns.set(style="whitegrid")
    data = load_horizontal_with_imputations()
    data = data[data['#removed'] < 0.95]
    grid = sns.FacetGrid(data, hue='classifier', col='gamma', height=4)
    grid.map(sns.lineplot, '#removed', 'accuracy', alpha=.7)
    plt.xticks(np.arange(0, 1.0, 0.10), np.arange(0, 100, 10))
    plt.legend()

    robustness_with_imp = load_horizontal_with_imputation_robustness()
    robustness_with_imp['imputation'] = ['with imputation' for i in range(len(robustness_with_imp))]
    robustness = load_horizontal_robustness()
    robustness['imputation'] = ['without' for i in range(len(robustness))]
    robustness = robustness.append(robustness_with_imp).reset_index()

    robustness = robustness.rename(columns={'#removed': '%removed'})
    robustness = robustness[robustness['%removed'] < 0.95]
    robustness['gamma'] = robustness['gamma'].astype('int32')
    grid2 = sns.FacetGrid(robustness, hue='imputation', col='gamma', height=4)
    grid2.map(sns.lineplot, '%removed', 'false miss', alpha=.7)
    plt.xticks(np.arange(0, 1.0, 0.10), np.arange(0, 100, 10))
    plt.legend()
    plt.show()


if __name__ == '__main__':

   # plot_horizontal_attack()
    #plot_bit_flipping_attack()
    plot_transferable_feature_selection()
    #plot_horizontal_with_imputation()
