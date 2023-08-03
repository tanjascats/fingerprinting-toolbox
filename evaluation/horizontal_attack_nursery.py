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
    dataset = datasets.Nursery()
    # modify this to a class
    parameter_grid = {'fp_len': [32, 64, 128],
                      'gamma': [1, 1.11, 1.25, 1.43, 1.67, 2, 2.5, 3.33, 5, 10],
                      'xi': [1]}  # 30 combinations
                      # frequency of marks (100%, 90%, 80%, 70%, 60%, 50%, 40%, 30%, 20%, 10%) -> sometimes it needs more granularity towards small percentages e
                      # 'xi': [1, 2, 4]}  # 90 combinations -> irrelevant for marking categorical data
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
                    with open('fingerprinted_data/nursery/nursery_l{}_g{}_x{}_{}_4.csv'.format(
                            fp_len, gamma, xi, secret_key), 'wb') as outfile:
                        fp_dataset.dataframe.to_csv(outfile, index=False)


def horizontal_attack(overwrite_existing=False): # prerequisite is that the fingerprinted datasets are available fingerprinted_data/breast_cancer_w
    # modify this to a class
    # parameter_grid = {'fp_len': [32, 64, 128],
    #                   'gamma': [1, 1.11, 1.25, 1.43, 1.67, 2, 2.5, 3.33, 5, 10],
    #                   # frequency of marks (100%, 90%, 80%, 70%, 60%, 50%, 40%, 30%, 20%, 10%) -> sometimes it needs more granularity towards small percentages e
    #                   'xi': [1, 2, 4]}  # 90 combinations
    baseline = 100

    # read existing experiments
    all_experiment_results = os.listdir('robustness/horizontal/nursery')
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
    all_fp_datasets = os.listdir('fingerprinted_data/nursery')
    for fp_dataset_path in all_fp_datasets:
        fp_dataset = datasets.Dataset(path='fingerprinted_data/nursery/' + fp_dataset_path,
                                      target_attribute='target', primary_key_attribute='Id')
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
        strength_grid = np.arange(0.1, 1.1, 0.1)
        strength_grid = [round(1.0 - s, 1) for s in strength_grid] # we reverse strength grid to speed up the experiment
        false_miss = dict()
        misattribution = dict()
        false_miss[0] = 0
        misattribution[0] = 0
        for strength in strength_grid:
            false_miss[strength] = 0;   misattribution[strength] = 0
            # attack x100
            for i in range(100):
                attack = attacks.HorizontalSubsetAttack()
                attacked_fp_dataset = attack.run(fp_dataset.dataframe, strength=strength,
                                                 random_state=i*int(strength*100))
                attacked_fp_dataset = datasets.Dataset(dataframe=attacked_fp_dataset, target_attribute='target',
                                                       primary_key_attribute='Id')
                suspect = scheme.detection(attacked_fp_dataset, secret_key=secret_key)
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
        with open('robustness/horizontal/nursery/false_miss_l{}_g{}_x{}.json'.format(fp_len, gamma, xi), 'w') as outfile:
            json.dump(false_miss, outfile)
        modified_files.append('robustness/horizontal/nursery/false_miss_l{}_g{}_x{}.json'.format(fp_len, gamma, xi))
        with open('robustness/horizontal/nursery/misattribution_l{}_g{}_x{}.json'.format(fp_len, gamma, xi), 'w') as outfile:
            json.dump(misattribution, outfile)
        modified_files.append('robustness/horizontal/nursery/misattribution_l{}_g{}_x{}.json'.format(fp_len, gamma, xi))

    # log the run
    timestamp = time.ctime()
    run_log = {'time': timestamp,
               'dataset': 'nursery',
               'fingerprinted_datasets': all_fp_datasets,
               'scheme': 'universal',
               'attack': 'horizontal subset',
               'modified files': modified_files}
    with open('robustness/run_logs/run_log_{}.json'.format(str(timestamp.replace(' ', '-').replace(':','-'))), 'w') as outfile:
        json.dump(run_log, outfile)


def horizontal_check():
    fp_dataset = datasets.Dataset(path='fingerprinted_data/nursery/nursery_l32_g1_x1_4370315727_4.csv',
                                      target_attribute='target', primary_key_attribute='Id')

    # sanity check
    scheme = Universal(fingerprint_bit_length=32, gamma=1, xi=1)
    suspect = scheme.detection(fp_dataset, secret_key=4370315727)
    if suspect != 4:
        #baseline -= 1
         # this line should not print !
        print('Detection went wrong: parameters {},{},{} ......................'.format(32, 1, 1))
    attack = attacks.HorizontalSubsetAttack()
    attacked_fp_dataset = attack.run(fp_dataset.dataframe, strength=0.999, random_state=1).sort_index()
    print(attacked_fp_dataset)
    print(fp_dataset.dataframe)
    attacked_fp_dataset = datasets.Dataset(dataframe=attacked_fp_dataset, target_attribute='target',
                                           primary_key_attribute='Id')
    suspect = scheme.detection(attacked_fp_dataset, secret_key=4370315727)


def main():
    horizontal_attack()


if __name__ == '__main__':
    main()
