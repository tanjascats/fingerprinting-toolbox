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
import time


def vertical_attack(overwrite_existing=False): # prerequisite is that the fingerprinted datasets are available fingerprinted_data/breastcancer
    # modify this to a class
    # parameter_grid = {'fp_len': [32, 64, 128],
    #                   'gamma': [1, 1.11, 1.25, 1.43, 1.67, 2, 2.5, 3.33, 5, 10],
    #                   # frequency of marks (100%, 90%, 80%, 70%, 60%, 50%, 40%, 30%, 20%, 10%) -> sometimes it needs more granularity towards small percentages e
    #                   'xi': [1, 2, 4]}  # 90 combinations
    baseline = 100

    # read existing experiments
    all_experiment_results = os.listdir('robustness/vertical/breastcancer')
    existing_results = []
    for exp_path in all_experiment_results:
        file_name = exp_path.split('_')
        existing_xi = int(file_name[-1][1:-5])
        existing_gamma = float(file_name[-2][1:])
        existing_fp_len = int(file_name[-3][1:])
        measure = file_name[-4]
        if not measure == 'miss':
            continue
        existing_results.append([existing_fp_len, existing_gamma, existing_xi])

    # for logging
    modified_files = []

    # grid search
    # read all fingerprinted datasets
    columns = ['age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat']


    all_fp_datasets = os.listdir('fingerprinted_data/breastcancer')
    for fp_dataset_path in all_fp_datasets:
        fp_dataset = datasets.Dataset(path='fingerprinted_data/breastcancer/' + fp_dataset_path,
                                      target_attribute='recurrence', primary_key_attribute='Id')
        a, fp_len, gamma, xi, secret_key, r = fp_dataset_path.split('_')
        fp_len = int(fp_len[1:]); gamma = float(gamma[1:]); xi = int(xi[1:]); secret_key = int(secret_key)
        if xi == 2 or xi == 4: continue # skip multiple values for xi because xi does not affect the subset attack

        # skip existing experiments if overwriting flag is not raises
        if not overwrite_existing:
            if [fp_len, gamma, xi] in existing_results:
                continue

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
            strength = len(columns) - strength # we reverse the order of strength to speed up the experiments
            false_miss[strength] = 0;   misattribution[strength] = 0
            # attack 100x
            for i in range(100):
                attack = attacks.VerticalSubsetAttack()
                attacked_fp_dataset = attack.run_random(fp_dataset.dataframe, number_of_columns=strength,
                                                        target_attr='recurrence', primary_key='Id',
                                                        random_state=i*strength)
                attacked_fp_dataset = datasets.Dataset(dataframe=attacked_fp_dataset, target_attribute='recurrence',
                                                       primary_key_attribute='Id')
                suspect = scheme.detection(attacked_fp_dataset, secret_key=secret_key, original_attributes=columns,
                                           target_attribute='recurrence', primary_key_attribute='Id')
                if suspect != 4:
                    false_miss[strength] += 1
                    if suspect != -1:
                        misattribution[strength] += 1
            false_miss[strength] /= 100
            misattribution[strength] /= 100
            # early stop criteria
            if false_miss[strength] == 0.0:
                break
        print(false_miss)
        print(misattribution)
        with open('robustness/vertical/breastcancer/false_miss_l{}_g{}_x{}.json'.format(fp_len, gamma, xi), 'w') as outfile:
            json.dump(false_miss, outfile)
        modified_files.append('robustness/vertical/breastcancer/false_miss_l{}_g{}_x{}.json'.format(fp_len, gamma, xi))
        with open('robustness/vertical/breastcancer/misattribution_l{}_g{}_x{}.json'.format(fp_len, gamma, xi), 'w') as outfile:
            json.dump(misattribution, outfile)
        modified_files.append('robustness/vertical/breastcancer/misattribution_l{}_g{}_x{}.json'.format(fp_len, gamma, xi))

    # log the run
    timestamp = time.ctime()
    run_log = {'time': timestamp,
               'dataset': 'breastcancer',
               'fingerprinted_datasets': all_fp_datasets,
               'scheme': 'universal',
               'attack': 'vertical subset',
               'modified files': modified_files}
    with open('robustness/run_logs/run_log_{}.json'.format(timestamp.replace(' ', '-').replace(':','-')), 'w') as outfile:
        json.dump(run_log, outfile)


def vertical_check():
    fp_dataset = datasets.Dataset(path='fingerprinted_data/breastcancer/breastcancer_l32_g1_x1_4370315727_4.csv',
                                      target_attribute='recurrence', primary_key_attribute='Id')
    original_attributes = fp_dataset.columns.drop(['recurrence', 'Id'])
    print(original_attributes)
    # sanity check
    scheme = Universal(fingerprint_bit_length=32, gamma=1, xi=1)
    suspect = scheme.detection(fp_dataset, secret_key=4370315727, target_attribute='recurrence')
    if suspect != 4:
        #baseline -= 1
         # this line should not print !
        print('Detection went wrong: parameters {},{},{} ......................'.format(32, 1, 1))
    attack = attacks.VerticalSubsetAttack()
    #attacked_fp_dataset = attack.run_random(dataset=fp_dataset.dataframe, number_of_columns=3,
    #                                        target_attr='recurrence', primary_key='Id')
    attacked_fp_dataset = attack.run(dataset=fp_dataset.dataframe, columns=['tumor-size','inv-nodes'])
    print(attacked_fp_dataset)
    print(fp_dataset.dataframe)
    attacked_fp_dataset = datasets.Dataset(dataframe=attacked_fp_dataset, target_attribute='recurrence', primary_key_attribute='Id')
    suspect = scheme.detection(attacked_fp_dataset, secret_key=4370315727, original_attributes=original_attributes,
                               target_attribute='recurrence', primary_key_attribute='Id')


def vertical_false_miss_estimation():
    dataset = datasets.BreastCancer()
    parameter_grid = {'fp_len': [32, 64, 128],
                      'gamma': [1, 1.11, 1.25, 1.43, 1.67, 2, 2.5, 3.33, 5, 10]}
    for fp_len in parameter_grid['fp_len']:
        for gamma in parameter_grid['gamma']:
            scheme = Universal(fingerprint_bit_length=fp_len, gamma=gamma)
            false_miss = dict()
            for strength in np.arange(0.0, 1.1, 0.1):
                attack = attacks.VerticalSubsetAttack()
                false_miss[strength] = attack.false_miss_estimation(dataset=dataset, strength_rel=strength, scheme=scheme)
            with open('robustness/vertical_est/breastcancer/false_miss_l{}_g{}_x1.json'.format(fp_len, gamma),
                      'w') as outfile:
                json.dump(false_miss, outfile)


def main():
    vertical_false_miss_estimation()


if __name__ == '__main__':
    main()