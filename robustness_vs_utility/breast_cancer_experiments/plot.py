from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from robustness_vs_utility.breast_cancer_experiments.log_formatted import experiment_results_data
import pickle
import numpy as np
from plotnine import *
from plotnine import ggplot


def load_data():
    data = pd.DataFrame(columns=['#removed', 'removed_attr', 'accuracy', 'gamma', 'classifier'])
    for classifier in experiment_results_data:
        data_dict_1 = experiment_results_data[classifier][0]
        for key in data_dict_1:
            for attr_comb in data_dict_1[key]:
                data.loc[len(data)] = [key, attr_comb, data_dict_1[key][attr_comb], 1, classifier]

        data_dict_2 = experiment_results_data[classifier][1]
        for key in data_dict_2:
            for attr_comb in data_dict_2[key]:
                data.loc[len(data)] = [key, attr_comb, data_dict_2[key][attr_comb], 2, classifier]

        data_dict_3 = experiment_results_data[classifier][2]
        for key in data_dict_3:
            for attr_comb in data_dict_3[key]:
                data.loc[len(data)] = [key, attr_comb, data_dict_3[key][attr_comb], 3, classifier]

        data_dict_5 = experiment_results_data[classifier][3]
        for key in data_dict_5:
            for attr_comb in data_dict_5[key]:
                data.loc[len(data)] = [key, attr_comb, data_dict_5[key][attr_comb], 5, classifier]
        print(data.head())

    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.7168, 'gamma': 1,
                           'classifier': 'decision tree'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.7341888341543513, 'gamma': 2,
                           'classifier': 'decision tree'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.7321714285714285, 'gamma': 3,
                           'classifier': 'decision tree'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.7485714285714286, 'gamma': 5,
                           'classifier': 'decision tree'}
    # copy
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.7168, 'gamma': 1,
                           'classifier': 'decision tree'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.7341888341543513, 'gamma': 2,
                           'classifier': 'decision tree'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.7321714285714285, 'gamma': 3,
                           'classifier': 'decision tree'}
    # full fingerprinted dataset
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.7485714285714286, 'gamma': 5,
                           'classifier': 'decision tree'}

    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.6915517241379311, 'gamma': 1,
                           'classifier': 'gradient boosting'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.6949564860426929, 'gamma': 2,
                           'classifier': 'gradient boosting'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.7031206896551722, 'gamma': 3,
                           'classifier': 'gradient boosting'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.6791346469622331, 'gamma': 5,
                           'classifier': 'gradient boosting'}
    # copy
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.6915517241379311, 'gamma': 1,
                           'classifier': 'gradient boosting'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.6949564860426929, 'gamma': 2,
                           'classifier': 'gradient boosting'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.7031206896551722, 'gamma': 3,
                           'classifier': 'gradient boosting'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.6791346469622331, 'gamma': 5,
                           'classifier': 'gradient boosting'}

    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.686536945812808, 'gamma': 1,
                           'classifier': 'knn'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy':  0.6752128078817734, 'gamma': 2,
                           'classifier': 'knn'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy':  0.672199671592775, 'gamma': 3,
                           'classifier': 'knn'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.7099, 'gamma': 5,
                           'classifier': 'knn'}
    # copy
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.686536945812808, 'gamma': 1,
                           'classifier': 'knn'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.6752128078817734, 'gamma': 2,
                           'classifier': 'knn'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.672199671592775, 'gamma': 3,
                           'classifier': 'knn'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.7099, 'gamma': 5,
                           'classifier': 'knn'}

    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.7110714285714285, 'gamma': 1,
                           'classifier': 'logistic regression'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.716390804597701, 'gamma': 2,
                           'classifier': 'logistic regression'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.7210467980295566, 'gamma': 3,
                           'classifier': 'logistic regression'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.7207027914614121, 'gamma': 5,
                           'classifier': 'logistic regression'}
    # copy
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.7110714285714285, 'gamma': 1,
                           'classifier': 'logistic regression'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.716390804597701, 'gamma': 2,
                           'classifier': 'logistic regression'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.7210467980295566, 'gamma': 3,
                           'classifier': 'logistic regression'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.7207027914614121, 'gamma': 5,
                           'classifier': 'logistic regression'}

    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.6913526272577997, 'gamma': 1,
                           'classifier': 'svm'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.6916546798029557, 'gamma': 2,
                           'classifier': 'svm'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.6904728243021346, 'gamma': 3,
                           'classifier': 'svm'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.6942472085385878, 'gamma': 5,
                           'classifier': 'svm'}
    # copy
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.6913526272577997, 'gamma': 1,
                           'classifier': 'svm'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.6916546798029557, 'gamma': 2,
                           'classifier': 'svm'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.6904728243021346, 'gamma': 3,
                           'classifier': 'svm'}
    data.loc[len(data)] = {'#removed': 0, 'removed_attr': None, 'accuracy': 0.6942472085385878, 'gamma': 5,
                           'classifier': 'svm'}

    return data


def load_vertical_robustness():
    n_exp = 1000
    robustness = [[], [], [], []]
    results = pd.DataFrame(columns=['#removed', 'false miss', 'gamma', 'gamma_id'])
    # !!!! RESULTS FOR BREAST CANCER L=8!!!!
    # gamma = 1
    # brojevi -detected fingerprints od 0 uklonjenih do 8 uklonjenih
    robustness[0] = n_exp - np.array([1000, 1000, 1000, 980, 993, 993, 958, 942, 919])
    robustness[0] = [i/1000 for i in robustness[0]]
    for r in range(len(robustness[0])):
        results.loc[len(results)] = {'#removed': r, 'false miss': robustness[0][r], 'gamma': 1, 'gamma_id': 0}
    # ----------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------- #
    # gamma = 2
    robustness[1] = n_exp - np.array([1000, 1000, 1000, 973, 957, 931, 831, 731, 734])
    robustness[1] = [i / 1000 for i in robustness[1]]
    for r in range(len(robustness[1])):
        results.loc[len(results)] = {'#removed': r, 'false miss': robustness[1][r], 'gamma': 2, 'gamma_id': 1}
    # ----------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------- #
    # gamma = 3
    robustness[2] = n_exp - np.array([1000, 996, 977, 922, 884, 841, 693, 606, 562])
    robustness[2] = [i / 1000 for i in robustness[2]]
    for r in range(len(robustness[2])):
        results.loc[len(results)] = {'#removed': r, 'false miss': robustness[2][r], 'gamma': 3, 'gamma_id': 2}
    # ----------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------- #
    # gamma = 5
    robustness[3] = n_exp - np.array([1000, 964, 886, 727, 614, 474, 455, 381, 257])
    robustness[3] = [i / 1000 for i in robustness[3]]
    for r in range(len(robustness[3])):
        results.loc[len(results)] = {'#removed': r, 'false miss': robustness[3][r], 'gamma': 5, 'gamma_id': 3}

    return results


def plot():
    sns.set(style="whitegrid")
    data = load_data()
    robustness = load_vertical_robustness()
    robustness['gamma'] = robustness['gamma'].astype('int32')
    robustness['#removed'] = robustness['#removed'].astype('int32')
    data['#removed'] = data['#removed'].astype('int32')
    # manipulate a bit

    # gamma removed false miss
    grid1 = sns.FacetGrid(data, hue='classifier', col='gamma', height=4)
    grid1.map(sns.lineplot, '#removed', 'accuracy', alpha=.7)
    plt.xticks(np.arange(0, 9, step=1))
    plt.legend()

    grid2 = sns.FacetGrid(robustness, col='gamma', height=2.5)
    grid2.map(sns.lineplot, '#removed', 'false miss', alpha=.7)
    plt.xticks(np.arange(0, 9, step=1))
    plt.show()

    # robustness
    #x = np.array([i for i in range(8)])
    #y_gamma_5 = (1000 - np.flip(np.array([964, 886, 727, 614, 474, 455, 381, 257, 0]))) / 100
    #sns.lineplot(x=x, y=y_gamma_5, label='$\gamma$ = 5')


def demo():
    sns.set(style="ticks")
    tips = sns.load_dataset("tips")
    print(tips.head())
    g = sns.FacetGrid(tips, hue="time", col="sex", height=4)
    g.map(plt.scatter, "total_bill", "tip", alpha=.7)
    g.add_legend()
    plt.show()


def load_horizontal_attack():
    classifiers = ['svm', 'decision_tree', 'gradient_boosting', 'logistic_regression', 'knn']
    gammae = [1, 2, 3, 5]
    results = pd.DataFrame()
    for classifier in classifiers:
        for gamma in gammae:
            try:
                results_intermed = pickle.load(open('robustness_vs_utility/breast_cancer_experiments/'
                                                    'horizontal_' + classifier + "_" + str(gamma), 'rb'))
            except FileNotFoundError:
                continue
            results = results.append(results_intermed, ignore_index=True)
    # append from all files
    return results


def load_horizontal_robustness():
    n_exp = 1000
    robustness = [[], [], [], []]
    results = pd.DataFrame(columns=['#removed', 'false miss', 'gamma', 'gamma_id'])
    # !!!! RESULTS FOR BREAST CANCER L=8!!!!
    # gamma = 1
    robustness[0] = n_exp - np.flip(np.array([135, 381, 648, 786, 897, 955, 978, 993, 999, 1000, 999, 1000,
                                              1000, 1000, 1000, 1000, 1000, 1000, 1000]))
    robustness[0] = [i / 1000 for i in robustness[0]]
    for r in range(len(robustness[0])):
        results.loc[len(results)] = {'#removed': r* 0.05, 'false miss': robustness[0][r], 'gamma': 1, 'gamma_id': 0}
    # -------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------- #
    # gamma = 2
    robustness[1] = n_exp - np.flip(np.array([17, 115, 223, 460, 584, 720, 777, 876, 915, 954, 961, 991, 988,
                                              992, 999, 1000, 1000, 1000, 1000]))
    robustness[1] = [i / 1000 for i in robustness[1]]
    for r in range(len(robustness[1])):
        results.loc[len(results)] = {'#removed': r*0.05, 'false miss': robustness[1][r], 'gamma': 2, 'gamma_id': 1}
    # ----------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------- #
    # gamma = 3
    robustness[2] = n_exp - np.flip(np.array([9, 28, 119, 203, 360, 492, 608, 676, 777, 833, 890, 904, 952,
                                              972, 990, 996, 998, 1000, 1000]))
    robustness[2] = [i / 1000 for i in robustness[2]]
    for r in range(len(robustness[2])):
        results.loc[len(results)] = {'#removed': r*0.05, 'false miss': robustness[2][r], 'gamma': 3, 'gamma_id': 2}
    # ----------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------- #
    # gamma = 5
    robustness[3] = n_exp - np.flip(np.array([0, 3, 10, 52, 75, 186, 271, 345, 454, 577, 620, 677, 785, 820, 892,
                                              917, 961, 986, 1000]))
    robustness[3] = [i / 1000 for i in robustness[3]]
    for r in range(len(robustness[3])):
        results.loc[len(results)] = {'#removed': r*0.05, 'false miss': robustness[3][r], 'gamma': 5, 'gamma_id': 3}
    # ----------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------- #
    return results


def plot_horizontal_attack():
    sns.set(style="whitegrid")
    data = load_horizontal_attack()
    grid = sns.FacetGrid(data, hue='classifier', col='gamma', height=4)
    grid.map(sns.lineplot, '#removed', 'accuracy', alpha=.7)
    plt.xticks(np.arange(0, 1.0, 0.10), np.arange(0, 100, 10))
    plt.xlabel('%removed')
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
    gammae = [1, 2, 3, 5]
    results = pd.DataFrame()
    for classifier in classifiers:
        for gamma in gammae:
            try:
                results_intermed = pickle.load(open('robustness_vs_utility/breast_cancer_experiments/'
                                                    'bit-flip_' + classifier + "_" + str(gamma), 'rb'))
            except FileNotFoundError:
                print('Change configuration :)')
                continue
            results = results.append(results_intermed, ignore_index=True)
    # append from all files
    #results.loc[len(results)] = {'#removed': 0.6, 'accuracy': 0.674072, 'gamma': 5, 'gamma_id': 3}
    #results.loc[len(results)] = {'#removed': 0.6, 'accuracy': 0.685386, 'gamma': 5, 'gamma_id': 3}
    #results.loc[len(results)] = {'#removed': 0.6, 'accuracy': 0.702890, 'gamma': 5, 'gamma_id': 3}
    #results.loc[len(results)] = {'#removed': 0.6, 'accuracy': 0.692890, 'gamma': 5, 'gamma_id': 3}
    #results.loc[len(results)] = {'#removed': 0.6, 'accuracy': 0.671478, 'gamma': 5, 'gamma_id': 3}

    #results.loc[len(results)] = {'#removed': 0.6, 'accuracy': 0.710148, 'gamma': 3, 'gamma_id': 2}
    #results.loc[len(results)] = {'#removed': 0.6, 'accuracy': 0.664565, 'gamma': 3, 'gamma_id': 2}
    #results.loc[len(results)] = {'#removed': 0.6, 'accuracy': 0.678251, 'gamma': 3, 'gamma_id': 2}
    #results.loc[len(results)] = {'#removed': 0.6, 'accuracy': 0.691921, 'gamma': 3, 'gamma_id': 2}
    #results.loc[len(results)] = {'#removed': 0.6, 'accuracy': 0.674548, 'gamma': 3, 'gamma_id': 2}

    #results.loc[len(results)] = {'#removed': 0.6, 'accuracy': 0.692167, 'gamma': 1, 'gamma_id': 0}
    #results.loc[len(results)] = {'#removed': 0.6, 'accuracy': 0.713350, 'gamma': 1, 'gamma_id': 0}
    #results.loc[len(results)] = {'#removed': 0.6, 'accuracy': 0.705854, 'gamma': 1, 'gamma_id': 0}
    #results.loc[len(results)] = {'#removed': 0.6, 'accuracy': 0.671215, 'gamma': 1, 'gamma_id': 0}
    #results.loc[len(results)] = {'#removed': 0.6, 'accuracy': 0.692406, 'gamma': 1, 'gamma_id': 0}

    #results.loc[len(results)] = {'#removed': 0.6, 'accuracy': 0.682053, 'gamma': 2, 'gamma_id': 1}
    #results.loc[len(results)] = {'#removed': 0.6, 'accuracy': 0.699433, 'gamma': 2, 'gamma_id': 1}
    #results.loc[len(results)] = {'#removed': 0.6, 'accuracy': 0.681330, 'gamma': 2, 'gamma_id': 1}
    #results.loc[len(results)] = {'#removed': 0.6, 'accuracy': 0.681691, 'gamma': 2, 'gamma_id': 1}
    #results.loc[len(results)] = {'#removed': 0.6, 'accuracy': 0.663834, 'gamma': 2, 'gamma_id': 1}
    return results


def load_bit_flipping_robustness():
    n_exp = 1000
    robustness = [[], [], [], []]
    results = pd.DataFrame(columns=['#removed', 'false miss', 'gamma', 'gamma_id'])
    # !!!! RESULTS FOR BREAST CANCER L=8!!!!
    # --------------------------------------- #
    # gamma = 1
    # --------------------------------------- #
    robustness[0] = 1000 - np.array([1000, 1000, 1000, 1000, 1000, 1000, 999, 995, 974, 978, 958, 939])
    robustness[0] = [i / 1000 for i in robustness[0]]
    for r in range(len(robustness[0])):
        results.loc[len(results)] = {'#removed': r* 0.05, 'false miss': robustness[0][r], 'gamma': 1,
                                     'gamma_id': 0}
    # --------------------------------------- #
    # gamma = 2
    # --------------------------------------- #
    robustness[1] = 1000 - np.array([1000, 1000, 1000, 998, 978, 987, 974, 950, 869, 868, 758, 697])
    robustness[1] = [i / 1000 for i in robustness[1]]
    for r in range(len(robustness[1])):
        results.loc[len(results)] = {'#removed': r* 0.05, 'false miss': robustness[1][r], 'gamma': 2,
                                     'gamma_id': 1}
    # --------------------------------------- #
    # gamma = 3
    # --------------------------------------- #
    robustness[2] = 1000 - np.array([1000, 1000, 968, 994, 960, 896, 876, 848, 754, 679, 588, 496])
    robustness[2] = [i / 1000 for i in robustness[2]]
    for r in range(len(robustness[2])):
        results.loc[len(results)] = {'#removed': r * 0.05, 'false miss': robustness[2][r], 'gamma': 3,
                                     'gamma_id': 2}
    # --------------------------------------- #
    # gamma = 5
    # --------------------------------------- #
    robustness[3] = 1000 - np.array([1000, 979, 910, 848, 859, 704, 641, 545, 461, 376, 332, 277])
    robustness[3] = [i / 1000 for i in robustness[3]]
    for r in range(len(robustness[3])):
        results.loc[len(results)] = {'#removed': r * 0.05, 'false miss': robustness[3][r], 'gamma': 5,
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

    robustness = load_bit_flipping_robustness()
    robustness = robustness.rename(columns={'#removed': '%flipped'})
    robustness['gamma'] = robustness['gamma'].astype('int32')
    grid2 = sns.FacetGrid(robustness, col='gamma', height=2.5)
    grid2.map(sns.lineplot, '%flipped', 'false miss', alpha=.7)
    plt.xticks(np.arange(0, 0.6, 0.05), np.arange(0, 60, 5))
    plt.show()
    plt.show()


if __name__ == '__main__':
    #plot()
    #exit()
    #plot_horizontal_attack()
    plot_bit_flipping_attack()
    #demo()
