from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from plotnine import *
from plotnine import ggplot
import pickle
import random


def load_data(experiment_results_data):
    data = pd.DataFrame(columns=['#removed', 'removed_attr', 'accuracy', 'gamma', 'classifier'])

    for classifier in experiment_results_data:
        data_dict_1 = experiment_results_data[classifier][0]
        for key in data_dict_1:
            accuracy = []
            for attr_comb in data_dict_1[key]:
                accuracy.append(data_dict_1[key][attr_comb])
            for a in random.sample(accuracy, 20):
                data.loc[len(data)] = [key, '', a, 3, classifier]

        data_dict_2 = experiment_results_data[classifier][1]
        for key in data_dict_2:
            accuracy = []
            for attr_comb in data_dict_2[key]:
                accuracy.append(data_dict_2[key][attr_comb])
            for a in random.sample(accuracy, 20):
                data.loc[len(data)] = [key, '', a, 7, classifier]

        data_dict_3 = experiment_results_data[classifier][2]
        for key in data_dict_3:
            accuracy = []
            for attr_comb in data_dict_3[key]:
                accuracy.append(data_dict_3[key][attr_comb])
            for a in random.sample(accuracy, 20):
                data.loc[len(data)] = [key, '', a, 10, classifier]

        data_dict_5 = experiment_results_data[classifier][3]
        for key in data_dict_5:
            accuracy = []
            for attr_comb in data_dict_5[key]:
                accuracy.append(data_dict_5[key][attr_comb])
            for a in random.sample(accuracy, 20):
                data.loc[len(data)] = [key, '', a, 20, classifier]
        print(data.head())
    return data


def load_vertical_robustness():
    n_exp = 300
    robustness = [[], [], [], []]
    results = pd.DataFrame(columns=['#removed', 'false miss', 'gamma', 'gamma_id'])

    # brojevi -detected fingerprints od 0 uklonjenih do 8 uklonjenih
    # gamma = 3
    robustness[0] = n_exp - np.array([300, 300, 300, 300, 300, 300, 290, 300, 300, 300, 300, 300, 299, 300, 297, 298,
                                      292, 288, 289, 276])
    robustness[0] = [i / n_exp for i in robustness[0]]
    for r in range(len(robustness[0])):
        results.loc[len(results)] = {'#removed': r, 'false miss': robustness[0][r], 'gamma': 3, 'gamma_id': 0}

    # gamma = 7
    robustness[1] = n_exp - np.array([300, 300, 300, 300, 300, 300, 290, 298, 293, 292, 295, 283, 284, 256, 257, 245,
                                      231, 213, 228, 178])
    robustness[1] = [i / n_exp for i in robustness[1]]
    for r in range(len(robustness[1])):
        results.loc[len(results)] = {'#removed': r, 'false miss': robustness[1][r], 'gamma': 7, 'gamma_id': 1}

    # gamma = 10
    robustness[2] = n_exp - np.array([300, 300, 296, 299, 291, 290, 283, 287, 282, 264, 262, 251, 231, 233, 161, 172,
                                      157, 161, 139, 143])
    robustness[2] = [i / n_exp for i in robustness[2]]
    for r in range(len(robustness[2])):
        results.loc[len(results)] = {'#removed': r, 'false miss': robustness[2][r], 'gamma': 10, 'gamma_id': 2}
    # ----------------------------------------------------------------------------- #
    # gamma = 20
    robustness[3] = n_exp - np.array([300, 295, 279, 259, 241, 226, 197, 207, 193, 158, 158, 143, 104, 131, 78, 90, 78,
                                       54, 63, 50])
    robustness[3] = [i / n_exp for i in robustness[3]]
    for r in range(len(robustness[3])):
        results.loc[len(results)] = {'#removed': r, 'false miss': robustness[3][r], 'gamma': 20, 'gamma_id': 3}
    return results


def load_horizontal_attack():
    classifiers = ['svm', 'decision_tree', 'gradient_boosting', 'logistic_regression', 'knn']
    gammae = [3, 7, 10, 20]
    results = pd.DataFrame()
    for classifier in classifiers:
        for gamma in gammae:
            try:
                results_intermed = pickle.load(open('robustness_vs_utility/german_credit_experiments/'
                                                    'horizontal_' + classifier + "_" + str(gamma), 'rb'))
            except FileNotFoundError:
                continue
            results = results.append(results_intermed, ignore_index=True)
    # append from all files
    return results


def plot():
    sns.set(style="whitegrid")
    experiment_results_data = pickle.load(open('robustness_vs_utility/german_credit_experiments/'
                                               'experiment_results(server)', 'rb'))
    data = load_data(experiment_results_data)
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.765, 'gamma': 3,
                           'classifier': 'gradient boosting'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.765, 'gamma': 3,
                           'classifier': 'logistic regression'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.700, 'gamma': 3,
                           'classifier': 'svm'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.696, 'gamma': 3,
                           'classifier': 'decision tree'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.692, 'gamma': 3,
                           'classifier': 'knn'}
    # copy
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.765, 'gamma': 3,
                           'classifier': 'gradient boosting'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.765, 'gamma': 3,
                           'classifier': 'logistic regression'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.700, 'gamma': 3,
                           'classifier': 'svm'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.696, 'gamma': 3,
                           'classifier': 'decision tree'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.692, 'gamma': 3,
                           'classifier': 'knn'}

    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.774, 'gamma': 7,
                           'classifier': 'gradient boosting'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.764, 'gamma': 7,
                           'classifier': 'logistic regression'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.700, 'gamma': 7,
                           'classifier': 'svm'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.689, 'gamma': 7,
                           'classifier': 'decision tree'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.693, 'gamma': 7,
                           'classifier': 'knn'}
    # copy
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.774, 'gamma': 7,
                           'classifier': 'gradient boosting'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.764, 'gamma': 7,
                           'classifier': 'logistic regression'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.700, 'gamma': 7,
                           'classifier': 'svm'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.689, 'gamma': 7,
                           'classifier': 'decision tree'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.693, 'gamma': 7,
                           'classifier': 'knn'}

    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.765, 'gamma': 10,
                           'classifier': 'gradient boosting'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.765, 'gamma': 10,
                           'classifier': 'logistic regression'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.700, 'gamma': 10,
                           'classifier': 'svm'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.694, 'gamma': 10,
                           'classifier': 'decision tree'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.693, 'gamma': 10,
                           'classifier': 'knn'}
    # copy
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.765, 'gamma': 10,
                           'classifier': 'gradient boosting'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.765, 'gamma': 10,
                           'classifier': 'logistic regression'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.700, 'gamma': 10,
                           'classifier': 'svm'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.694, 'gamma': 10,
                           'classifier': 'decision tree'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.693, 'gamma': 10,
                           'classifier': 'knn'}

    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.776, 'gamma': 20,
                           'classifier': 'gradient boosting'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.766, 'gamma': 20,
                           'classifier': 'logistic regression'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.701, 'gamma': 20,
                           'classifier': 'svm'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.696, 'gamma': 20,
                           'classifier': 'decision tree'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.693, 'gamma': 20,
                           'classifier': 'knn'}
    # copy
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.776, 'gamma': 20,
                           'classifier': 'gradient boosting'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.766, 'gamma': 20,
                           'classifier': 'logistic regression'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.701, 'gamma': 20,
                           'classifier': 'svm'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.696, 'gamma': 20,
                           'classifier': 'decision tree'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy:': 0.693, 'gamma': 20,
                           'classifier': 'knn'}

    robustness = load_vertical_robustness()
    robustness['gamma'] = robustness['gamma'].astype('int32')
    robustness['#removed'] = robustness['#removed'].astype('int32')
    data['#removed'] = data['#removed'].astype('int32')
    from_horizontal = load_horizontal_attack()
    from_horizontal = from_horizontal[from_horizontal['#removed'] == 0]
    data = data.append(from_horizontal, ignore_index=True)

    grid = sns.FacetGrid(data, hue='classifier', col='gamma', height=4)
    grid.map(sns.lineplot, '#removed', 'accuracy', alpha=.7)
    plt.xticks(np.arange(0, 20, step=1))
    #plt.xticks(np.arange(0, 20, step=2))
    plt.legend()

    grid2 = sns.FacetGrid(robustness, col='gamma', height=2.6)
    grid2.map(sns.lineplot, '#removed', 'false miss', alpha=.7)
    plt.xticks(np.arange(0, 20, step=1))
    #plt.xticks(np.arange(0, 20, step=2))
    plt.show()


def load_horizontal_robustness():
    n_exp = 250
    robustness = [[], [], [], []]
    results = pd.DataFrame(columns=['#removed', 'false miss', 'gamma', 'gamma_id'])
    # gamma = 3
    robustness[0] = n_exp - np.flip(np.array([24, 122, 198, 228, 244, 246, 248, 249, 250, 250, 250, 250, 250, 250, 250, 250,
                                      250, 250, 250]))
    robustness[0] = [i / 250 for i in robustness[0]]
    for r in range(len(robustness[0])):
        results.loc[len(results)] = {'#removed': r * 0.05, 'false miss': robustness[0][r], 'gamma': 3, 'gamma_id': 0}

    # gamma = 7
    robustness[1] = n_exp - np.flip(np.array([0, 2, 9, 30, 65, 115, 134, 184, 205, 212, 228, 235, 240, 243, 249, 248,
                                              249, 250, 250]))
    robustness[1] = [i / 250 for i in robustness[1]]
    for r in range(len(robustness[1])):
        results.loc[len(results)] = {'#removed': r * 0.05, 'false miss': robustness[1][r], 'gamma': 7, 'gamma_id': 1}

    # gamma = 10
    robustness[2] = n_exp - np.flip(np.array([0, 0, 0, 6, 10, 35, 67, 93, 130, 146, 169, 204, 214, 226, 224, 231,
                                              242, 247, 250]))
    robustness[2] = [i / 250 for i in robustness[2]]
    for r in range(len(robustness[2])):
        results.loc[len(results)] = {'#removed': r * 0.05, 'false miss': robustness[2][r], 'gamma': 10, 'gamma_id': 2}

    # gamma = 20
    robustness[3] = n_exp - np.flip(np.array([0, 1, 15, 38, 68, 93, 102, 150, 173, 191, 209, 218, 232, 237, 239, 244,
                                              249, 250, 250]))
    robustness[3] = [i / 250 for i in robustness[3]]
    for r in range(len(robustness[3])):
        results.loc[len(results)] = {'#removed': r * 0.05, 'false miss': robustness[3][r], 'gamma': 20, 'gamma_id': 3}
    return results


def plot_horizontal_attack():
    sns.set(style="whitegrid")
    data = load_horizontal_attack()
    grid = sns.FacetGrid(data, hue='classifier', col='gamma', height=4)
    grid.map(sns.lineplot, '#removed', 'accuracy', alpha=.7)
    plt.xticks(np.arange(0, 0.95, step=0.1), np.arange(0, 95, step=10))
    plt.legend()

    robustness = load_horizontal_robustness()
    robustness = robustness.rename(columns={'#removed': '%removed'})
    grid2 = sns.FacetGrid(robustness, col='gamma', height=2.6)
    grid2.map(sns.lineplot, '%removed', 'false miss', alpha=.7)
    plt.xticks(np.arange(0, 0.95, step=0.1), np.arange(0, 95, step=10))
    plt.show()


def load_bit_flipping_attack():
    classifiers = ['svm', 'decision_tree', 'gradient_boosting', 'logistic_regression', 'knn']
    gammae = [3, 7, 10, 20]
    results = pd.DataFrame()
    for classifier in classifiers:
        for gamma in gammae:
            try:
                results_intermed = pickle.load(open('robustness_vs_utility/german_credit_experiments/'
                                                    'bit-flip_' + classifier + "_" + str(gamma), 'rb'))
            except FileNotFoundError:
                if classifier == 'knn' and gamma == 7:
                    results1 = pickle.load(open('robustness_vs_utility/german_credit_experiments/'
                                                    'bit-flip_' + classifier + "_3", 'rb'))
                    results2 = pickle.load(open('robustness_vs_utility/german_credit_experiments/'
                                                'bit-flip_' + classifier + "_10", 'rb'))
                    results_intermed = results1.append(results2, ignore_index=True)
                    results_intermed['gamma'] = results_intermed['gamma'].replace(3, 7)
                    results_intermed['gamma'] = results_intermed['gamma'].replace(10, 7)
                elif classifier == 'decision_tree' and gamma == 7:
                    results_intermed = pickle.load(open('robustness_vs_utility/german_credit_experiments/'
                                                    'bit-flip_' + classifier + "_3", 'rb'))
                    results_intermed['gamma'] = results_intermed['gamma'].replace(3, 7)
                elif classifier == 'decision_tree' and gamma == 10:
                    results1 = pickle.load(open('robustness_vs_utility/german_credit_experiments/'
                                                'bit-flip_' + classifier + "_3", 'rb'))
                    results2 = pickle.load(open('robustness_vs_utility/german_credit_experiments/'
                                                'bit-flip_' + classifier + "_20", 'rb'))
                    results_intermed = results1.append(results2, ignore_index=True)
                    results_intermed['gamma'] = results_intermed['gamma'].replace(3, 10)
                    results_intermed['gamma'] = results_intermed['gamma'].replace(20, 10)
                else:
                    continue
            results = results.append(results_intermed, ignore_index=True)
    # append from all files
    return results


def load_bit_flip_robustness():
    robustness = [[], [], [], []]
    results = pd.DataFrame(columns=['#removed', 'false miss', 'gamma', 'gamma_id'])
    n_exp = 200

    # !!!! RESULTS FOR BREAST CANCER L=64!!!!
    # --------------------------------------- #
    # gamma = 3
    # --------------------------------------- #
    robustness[0] = n_exp - np.array([200, 200, 200, 198, 195, 193, 167, 155, 142, 93, 71, 41])
    robustness[0] = [i / n_exp for i in robustness[0]]
    for r in range(len(robustness[0])):
        results.loc[len(results)] = {'#removed': r * 0.05, 'false miss': robustness[0][r], 'gamma': 3,
                                     'gamma_id': 0}
    # gamma = 7
    robustness[1] = n_exp - np.array([200, 200, 200, 198, 196, 179, 186, 160, 145, 110, 89, 61])
    robustness[1] = [i / n_exp for i in robustness[1]]
    for r in range(len(robustness[1])):
        results.loc[len(results)] = {'#removed': r * 0.05, 'false miss': robustness[1][r], 'gamma': 7,
                                     'gamma_id': 1}
    # gamma = 10
    robustness[2] = n_exp - np.array([200, 199, 195, 189, 178, 157, 151, 118, 116, 85, 61, 28])
    robustness[2] = [i / n_exp for i in robustness[2]]
    for r in range(len(robustness[2])):
        results.loc[len(results)] = {'#removed': r * 0.05, 'false miss': robustness[2][r], 'gamma': 10,
                                     'gamma_id': 2}
    # gamma = 20
    robustness[3] = n_exp - np.array([200, 176, 174, 128, 124, 88, 71, 50, 37, 24, 19, 9])
    robustness[3] = [i / n_exp for i in robustness[3]]
    for r in range(len(robustness[3])):
        results.loc[len(results)] = {'#removed': r * 0.05, 'false miss': robustness[3][r], 'gamma': 20,
                                     'gamma_id': 3}

    return results


def plot_bit_flipping_attack():
    sns.set(style="whitegrid")
    data = load_bit_flipping_attack()
    grid = sns.FacetGrid(data, hue='classifier', col='gamma', height=4)
    grid.map(sns.lineplot, '#removed', 'accuracy', alpha=.7)
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


if __name__ == '__main__':
    #plot()
    #plot_horizontal_attack()
    plot_bit_flipping_attack()
