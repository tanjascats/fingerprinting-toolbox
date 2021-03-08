import pandas as pd
import matplotlib.pyplot as plt
import pickle


def main():
    with open('adult_hsubset.df', 'rb') as f:
        forest_hsubset = pickle.load(f)
    with open('adult_flipping.df', 'rb') as f:
        forest_flipping = pickle.load(f)

    plt.plot(forest_hsubset['%marks'], forest_hsubset['robustness'], label='Subset attack', marker='o', markersize=4)
    plt.plot(forest_flipping['%marks'], forest_flipping['robustness'], label='Bit-flipping attack', marker='o',
             markersize=4)

    plt.title('Robustness against attacks')
    plt.xlabel('Number of fingerprint marks (% of data rows marked)')
    plt.ylabel('Robustness(%)')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
