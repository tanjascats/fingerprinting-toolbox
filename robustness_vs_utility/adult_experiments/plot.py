import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd
import numpy as np


def plot():
    data = pd.DataFrame(columns=['#removed', 'removed_attr', 'accuracy', 'gamma', 'classifier'])
    decision_tree5 = pickle.load(open('robustness_vs_utility/adult_experiments/decision_tree5', 'rb'))
    for key in decision_tree5:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(decision_tree5[key]),
                               'gamma': 5, 'classifier': 'decision tree'}
    decision_tree10 = pickle.load(open('robustness_vs_utility/adult_experiments/decision_tree10', 'rb'))
    for key in decision_tree10:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(decision_tree10[key]),
                               'gamma': 10, 'classifier': 'decision tree'}
    decision_tree20 = pickle.load(open('robustness_vs_utility/adult_experiments/decision_tree20', 'rb'))
    for key in decision_tree20:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(decision_tree20[key]),
                               'gamma': 20, 'classifier': 'decision tree'}
    decision_tree30 = pickle.load(open('robustness_vs_utility/adult_experiments/decision_tree30', 'rb'))
    for key in decision_tree30:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(decision_tree30[key]),
                               'gamma': 30, 'classifier': 'decision tree'}

    knn5 = pickle.load(open('robustness_vs_utility/adult_experiments/knn5', 'rb'))
    for key in knn5:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(knn5[key]),
                               'gamma': 5, 'classifier': 'knn'}
    knn10 = pickle.load(open('robustness_vs_utility/adult_experiments/knn10', 'rb'))
    for key in knn10:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(knn10[key]),
                               'gamma': 10, 'classifier': 'knn'}
    knn20 = pickle.load(open('robustness_vs_utility/adult_experiments/knn20', 'rb'))
    for key in knn20:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(knn20[key]),
                               'gamma': 20, 'classifier': 'knn'}
    knn30 = pickle.load(open('robustness_vs_utility/adult_experiments/knn30', 'rb'))
    for key in knn30:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(knn30[key]),
                               'gamma': 30, 'classifier': 'knn'}

    logistic_regression5 = pickle.load(open('robustness_vs_utility/adult_experiments/logistic_regression5', 'rb'))
    for key in logistic_regression5:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(logistic_regression5[key]),
                               'gamma': 5, 'classifier': 'logistic regression'}
    logistic_regression10 = pickle.load(open('robustness_vs_utility/adult_experiments/logistic_regression10', 'rb'))
    for key in logistic_regression10:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(logistic_regression10[key]),
                               'gamma': 10, 'classifier': 'logistic regression'}
    logistic_regression20 = pickle.load(open('robustness_vs_utility/adult_experiments/logistic_regression20', 'rb'))
    for key in logistic_regression20:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(logistic_regression20[key]),
                               'gamma': 20, 'classifier': 'logistic regression'}
    logistic_regression30 = pickle.load(open('robustness_vs_utility/adult_experiments/logistic_regression30', 'rb'))
    for key in logistic_regression30:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(logistic_regression30[key]),
                               'gamma': 30, 'classifier': 'logistic regression'}

    gradient_boosting5 = pickle.load(open('robustness_vs_utility/adult_experiments/gradient_boosting5', 'rb'))
    for key in gradient_boosting5:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(gradient_boosting5[key]),
                               'gamma': 5, 'classifier': 'gradient boosting'}
    gradient_boosting10 = pickle.load(open('robustness_vs_utility/adult_experiments/gradient_boosting10', 'rb'))
    for key in gradient_boosting10:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(gradient_boosting10[key]),
                               'gamma': 10, 'classifier': 'gradient boosting'}
    gradient_boosting20 = pickle.load(open('robustness_vs_utility/adult_experiments/gradient_boosting20', 'rb'))
    for key in gradient_boosting20:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(gradient_boosting20[key]),
                               'gamma': 20, 'classifier': 'gradient boosting'}
    gradient_boosting30 = pickle.load(open('robustness_vs_utility/adult_experiments/gradient_boosting30', 'rb'))
    for key in gradient_boosting30:
        data.loc[len(data)] = {'#removed': key, 'removed_attr': "?", 'accuracy': np.mean(gradient_boosting30[key]),
                               'gamma': 30, 'classifier': 'gradient boosting'}

    sns.set(style="ticks")
    grid = sns.FacetGrid(data, hue='classifier', col='gamma', height=4)
    grid.map(sns.lineplot, '#removed', 'accuracy', alpha=.7)
    grid.add_legend()
    plt.show()


def load_horizontal_attack():
    classifiers = ['svm', 'decision_tree', 'gradient_boosting', 'logistic_regression', 'knn']
    gammae = [5, 10, 20, 30]
    results = pd.DataFrame()
    for classifier in classifiers:
        for gamma in gammae:
            try:
                results_intermed = pickle.load(open('robustness_vs_utility/adult_experiments/'
                                                    'horizontal_' + classifier + "_" + str(gamma), 'rb'))
            except FileNotFoundError:
                continue
            results = results.append(results_intermed, ignore_index=True)
    # append from all files
    return results


def plot_horizontal_attack():
    sns.set(style="ticks")
    data = load_horizontal_attack()
    grid = sns.FacetGrid(data, hue='classifier', col='gamma', height=4)
    grid.map(sns.lineplot, '#removed', 'accuracy', alpha=.7)
    grid.add_legend()
    plt.show()


def load_bit_flipping_attack():
    classifiers = ['svm', 'decision_tree', 'gradient_boosting', 'logistic_regression', 'knn']
    gammae = [5, 10, 20, 30]
    results = pd.DataFrame()
    for classifier in classifiers:
        for gamma in gammae:
            try:
                results_intermed = pickle.load(open('robustness_vs_utility/adult_experiments/'
                                                    'bit-flip_' + classifier + "_" + str(gamma), 'rb'))
            except FileNotFoundError:
                continue
            results = results.append(results_intermed, ignore_index=True)
    # append from all files
    return results


def plot_bit_flipping_attack():
    sns.set(style="ticks")
    data = load_bit_flipping_attack()
    grid = sns.FacetGrid(data, hue='classifier', col='gamma', height=4)
    grid.map(sns.lineplot, '#removed', 'accuracy', alpha=.7)
    grid.add_legend()
    plt.show()


if __name__ == '__main__':
    plot_horizontal_attack()
    #plot_bit_flipping_attack()
    #demo()
