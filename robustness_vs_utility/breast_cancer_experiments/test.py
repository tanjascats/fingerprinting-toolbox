import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def single_feature_classification():
    data = pd.read_csv('datasets/breast_cancer_full.csv', index_col='Id')
    target = data.values[:, (len(data.columns) - 1)]
    model = DecisionTreeClassifier(criterion='entropy', max_depth=2)
    score = dict()
    for feature in data.columns[:-1]:
        feature_values = pd.get_dummies(data[feature])
        if len(feature_values.columns) == 2:
            feature_values = feature_values.drop(feature_values.columns[0], axis=1)
        score[feature] = np.mean(cross_val_score(model, feature_values, target, cv=10))
    return score


def plot_score(scores):
    print(scores)
    sns.barplot(x=list(scores.keys()), y=list(scores.values()))
    # baseline = 201 / (201+85) = 0.7027972027972028
    sns.lineplot(x=list(scores.keys()), y=0.7027972027972028)
    plt.show()


if __name__ == '__main__':
    scores = single_feature_classification()
    plot_score(scores)
