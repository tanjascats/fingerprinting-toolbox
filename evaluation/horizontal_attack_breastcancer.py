import json
import random
import time

import pandas as pd
from sklearn import datasets

import attacks
import datasets
from scheme import *
from attacks import *
import numpy as np
import os


def fingerprint_experiment_datasets():
    dataset = datasets.BreastCancer()
    # modify this to a class
    parameter_grid = {'fp_len': [32, 64],
                      'gamma': [1, 1.11, 1.25, 1.43, 1.67, 2, 2.5, 3.33, 5, 10],
                      'xi': [1]}  # everything is categorical, xi has no effect

    # grid search
    for fp_len in parameter_grid['fp_len']:
        for gamma in parameter_grid['gamma']:
            for xi in parameter_grid['xi']:
                scheme = Universal(fingerprint_bit_length=fp_len, gamma=gamma, xi=xi)
                secret_key = 4370315727
                fp_dataset = scheme.insertion(dataset=dataset, secret_key=secret_key, recipient_id=4)
                suspect = scheme.detection(fp_dataset, secret_key=secret_key)
                attempt = 0
                while suspect != 4 and attempt < 100:
                    # we need to find the fingerprint that works
                    secret_key += random.randint(-40, 40)
                    fp_dataset = scheme.insertion(dataset=dataset, secret_key=secret_key, recipient_id=4)
                    suspect = scheme.detection(fp_dataset, secret_key=secret_key)
                    attempt += 1
                if suspect != 4:
                    print('###################################################\n'
                          'THESE PARAMETERS CANNOT CREATE A ROBUST FINGERPRINT\n'
                          '###################################################')
                else:
                    # write to files
                    with open('fingerprinted_data/breastcancer/breastcancer_l{}_g{}_x{}_{}_4.csv'.format(
                            fp_len, gamma, xi, secret_key), 'wb') as outfile:
                        fp_dataset.dataframe.to_csv(outfile, index=False)


def main():
    fingerprint_experiment_datasets()


if __name__ == '__main__':
    main()