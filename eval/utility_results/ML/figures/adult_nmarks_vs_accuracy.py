import matplotlib.pyplot as plt
import pandas as pd
import pickle


def main():
    with open('../adult_GB.df', 'rb') as f:
        adult_GB = pickle.load(f)
    with open('../adult_knn.df', 'rb') as f:
        adult_knn = pickle.load(f)
    with open('../adult_LR.df', 'rb') as f:
        adult_LR = pickle.load(f)
    with open('../adult_RF.df', 'rb') as f:
        adult_RF = pickle.load(f)

    plt.plot(adult_GB['%marks'], adult_GB['delta accuracy'], label='Gradient Boosting', marker='o', markersize=4)
    plt.plot(adult_LR['%marks'], adult_LR['delta accuracy'], label='Logistic Regression', marker='o', markersize=4)
    plt.plot(adult_RF['%marks'], adult_RF['delta accuracy'], label='Random Forest', marker='o', markersize=4)
    plt.plot(adult_knn['%marks'], adult_knn['delta accuracy'], label='kNN', marker='o', markersize=4)

    plt.title('Utility effects on ML performance')
    plt.xlabel('Number of fingerprint marks (% of data rows marked)')
    plt.ylabel('$\Delta$ Accuracy')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
