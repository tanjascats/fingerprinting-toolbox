# ROBUSTNESS
# choose the scheme
# set the parameter grid
# fingerprint the data
# sanity check: detection rate of clean data
# apply attack x100
# try to detect
# record a false miss & misattribution
import json
import random

import pandas as pd
from sklearn import datasets

import attacks
import datasets
from scheme import *
from attacks import *
import numpy as np
import os

def vertical_attack(): # prerequisite is that the fingerprinted datasets are available fingerprinted_data/breast_cancer_w
    # modify this to a class
    # parameter_grid = {'fp_len': [32, 64, 128],
    #                   'gamma': [1, 1.11, 1.25, 1.43, 1.67, 2, 2.5, 3.33, 5, 10],
    #                   # frequency of marks (100%, 90%, 80%, 70%, 60%, 50%, 40%, 30%, 20%, 10%) -> sometimes it needs more granularity towards small percentages e
    #                   'xi': [1, 2, 4]}  # 90 combinations
    baseline = 100
    # grid search
    # read all fingerprinted datasets
    columns = ['clump-thickness','uniformity-of-cell-size','uniformity-of-cell-shape','marginal-adhesion',
               'single-epithelial-cell-size','bare-nuclei','bland-chromatin','normal-nucleoli', 'mitoses']

    all_fp_datasets = os.listdir('../fingerprinted_data/breast_cancer_w')
    for fp_dataset_path in all_fp_datasets:
        fp_dataset = datasets.Dataset(path='fingerprinted_data/breast_cancer_w/' + fp_dataset_path,
                                      target_attribute='class', primary_key_attribute='sample-code-number')
        a, b, c, fp_len, gamma, xi, secret_key, r = fp_dataset_path.split('_')
        fp_len = int(fp_len[1:]); gamma = float(gamma[1:]); xi = int(xi[1:]); secret_key = int(secret_key)
        if xi == 2 or xi == 4: continue # skip multiple values for xi because xi does not affect the subset attack
        # sanity check
        scheme = Universal(fingerprint_bit_length=fp_len, gamma=gamma, xi=xi)
        # suspect = scheme.detection(fp_dataset, secret_key=secret_key)
        # if suspect != 4:
        #     baseline -= 1
        #     # this line should not print !
        #     print('Detection went wrong: parameters {},{},{} ......................'.format(fp_len, gamma, xi))
        false_miss = dict()
        misattribution = dict()
        false_miss[0] = 0
        misattribution[0] = 0
        for strength in range(1, len(columns)):
            false_miss[strength] = 0;   misattribution[strength] = 0
            # attack 100x
            for i in range(100):
                attack = attacks.VerticalSubsetAttack()
                attacked_fp_dataset = attack.run_random(fp_dataset.dataframe, number_of_columns=strength,
                                                        target_attr='class', primary_key='sample-code-number',
                                                        random_state=i*strength)
                attacked_fp_dataset = datasets.Dataset(dataframe=attacked_fp_dataset, target_attribute='class',
                                                       primary_key_attribute='sample-code-number')
                suspect = scheme.detection(attacked_fp_dataset, secret_key=secret_key, original_attributes=columns,
                                           target_attribute='class', primary_key_attribute='sample-code-number')
                if suspect != 4:
                    false_miss[strength] += 1
                    if suspect != -1:
                        misattribution[strength] += 1
            false_miss[strength] /= 100
            misattribution[strength] /= 100
            # early stop criteria
            if false_miss[strength] == 1.0:
                break
        print(false_miss)
        print(misattribution)
        with open('robustness/vertical/breast_cancer_w/false_miss_l{}_g{}_x{}.json'.format(fp_len, gamma, xi), 'w') as outfile:
            json.dump(false_miss, outfile)
        with open('robustness/vertical/breast_cancer_w/misattribution_l{}_g{}_x{}.json'.format(fp_len, gamma, xi), 'w') as outfile:
            json.dump(misattribution, outfile)


def vertical_check():
    fp_dataset = datasets.Dataset(path='../fingerprinted_data/breast_cancer_w/breast_cancer_w_l32_g1_x1_4370315727_4.csv',
                                  target_attribute='class', primary_key_attribute='sample-code-number')
    original_attributes = fp_dataset.columns.drop(['class', 'sample-code-number'])
    print(original_attributes)
    # sanity check
    scheme = Universal(fingerprint_bit_length=32, gamma=1, xi=1)
    suspect = scheme.detection(fp_dataset, secret_key=4370315727, target_attribute='class')
    if suspect != 4:
        #baseline -= 1
         # this line should not print !
        print('Detection went wrong: parameters {},{},{} ......................'.format(32, 1, 1))
    attack = attacks.VerticalSubsetAttack()
    #attacked_fp_dataset = attack.run_random(dataset=fp_dataset.dataframe, number_of_columns=3,
    #                                        target_attr='class', primary_key='sample-code-number')
    attacked_fp_dataset = attack.run(dataset=fp_dataset.dataframe, columns=['uniformity-of-cell-size'])
    print(attacked_fp_dataset)
    print(fp_dataset.dataframe)
    attacked_fp_dataset = datasets.Dataset(dataframe=attacked_fp_dataset, target_attribute='class', primary_key_attribute='sample-code-number')
    suspect = scheme.detection(attacked_fp_dataset, secret_key=4370315727, original_attributes=original_attributes,
                               target_attribute='class', primary_key_attribute='sample-code-number')


def vertical_false_miss_estimation():
    dataset = datasets.BreastCancerWisconsin()
    parameter_grid = {'fp_len': [32, 64, 128],
                      'gamma': [1, 1.11, 1.25, 1.43, 1.67, 2, 2.5, 3.33, 5, 10]}
    for fp_len in parameter_grid['fp_len']:
        for gamma in parameter_grid['gamma']:
            scheme = Universal(fingerprint_bit_length=fp_len, gamma=gamma)
            false_miss = dict()
            for strength in np.arange(0.0, 1.1, 0.1):
                attack = attacks.VerticalSubsetAttack()
                false_miss[strength] = attack.false_miss_estimation(dataset=dataset, strength_rel=strength, scheme=scheme)
            with open('robustness/vertical_est/breast_cancer_w/false_miss_l{}_g{}_x1.json'.format(fp_len, gamma),
                      'w') as outfile:
                json.dump(false_miss, outfile)


def main():
    vertical_false_miss_estimation()


if __name__ == '__main__':
    main()