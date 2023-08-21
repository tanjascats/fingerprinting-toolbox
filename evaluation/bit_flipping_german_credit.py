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
import time

import attacks
import datasets
from scheme import *
from attacks import *
import numpy as np
import os
import vertical_attack_german_credit
import horizontal_attack_german_credit


def flipping_attack(overwrite_existing=False): # prerequisite is that the fingerprinted datasets are available fingerprinted_data/german_credit
    # read existing experiments
    all_experiment_results = os.listdir('robustness/flipping/german_credit')
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
    all_fp_datasets = os.listdir('fingerprinted_data/german_credit')
    for fp_dataset_path in all_fp_datasets:
        fp_dataset = datasets.Dataset(path='fingerprinted_data/german_credit/' + fp_dataset_path,
                                      target_attribute='target', primary_key_attribute='Id')
        a, b, fp_len, gamma, xi, secret_key, r = fp_dataset_path.split('_')
        fp_len = int(fp_len[1:]); gamma = float(gamma[1:]); xi = int(xi[1:]); secret_key = int(secret_key)
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
        #strength_grid = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        strength_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        false_miss = dict()
        misattribution = dict()
        false_miss[0] = 0
        misattribution[0] = 0
        for strength in strength_grid:
            false_miss[strength] = 0;   misattribution[strength] = 0
            # attack x100
            for i in range(100):
                attack = attacks.FlippingAttack()
                attacked_fp_dataset = attack.run(fp_dataset.dataframe, strength=strength,
                                                 random_state=i*int(strength*100), xi=xi)
                attacked_fp_dataset = datasets.Dataset(dataframe=attacked_fp_dataset, target_attribute='target',
                                                       primary_key_attribute='Id')
                suspect = scheme.detection(attacked_fp_dataset, secret_key=secret_key)
                if suspect != 4:
                    false_miss[strength] += 1
                    if suspect != -1:
                        misattribution[strength] += 1
            false_miss[strength] /= 100
            misattribution[strength] /= 100
            # --------------------- #
            # early stop criteria
            # IMPORTANT: early stop depends on whether attack strength is descending (0.0) or ascending (1.0)
            # --------------------- #
            if false_miss[strength] == 1.0:
                break
        print(false_miss)
        print(misattribution)
        with open('robustness/flipping/german_credit/false_miss_l{}_g{}_x{}.json'.format(fp_len, gamma, xi), 'w') as outfile:
            json.dump(false_miss, outfile)
        modified_files.append('robustness/horizontal/flipping/false_miss_l{}_g{}_x{}.json'.format(fp_len, gamma, xi))
        with open('robustness/flipping/german_credit/misattribution_l{}_g{}_x{}.json'.format(fp_len, gamma, xi), 'w') as outfile:
            json.dump(misattribution, outfile)
        modified_files.append('robustness/horizontal/flipping/misattribution_l{}_g{}_x{}.json'.format(fp_len, gamma, xi))

    # log the run
    timestamp = time.ctime()
    run_log = {'time': timestamp,
               'dataset': 'german_credit',
               'fingerprinted_datasets': all_fp_datasets,
               'scheme': 'universal',
               'attack': 'flipping subset',
               'modified files': modified_files}
    with open('robustness/run_logs/run_log_{}.json'.format(timestamp.replace(' ', '').replace(':','-')), 'w') as outfile:
        json.dump(run_log, outfile)


def flipping_check():
    fp_dataset = datasets.Dataset(path='fingerprinted_data/german_credit/german_credit_l32_g1_x1_4370315727_4.csv',
                                      target_attribute='target', primary_key_attribute='Id')

    # sanity check
    scheme = Universal(fingerprint_bit_length=32, gamma=1, xi=1)
    suspect = scheme.detection(fp_dataset, secret_key=4370315727)
    if suspect != 4:
        #baseline -= 1
         # this line should not print !
        print('Detection went wrong: parameters {},{},{} ......................'.format(32, 1, 1))
    attack = attacks.FlippingAttack()
    attacked_fp_dataset = attack.run(fp_dataset.dataframe, strength=0.5, random_state=1)
    print(attacked_fp_dataset)
    print(fp_dataset.dataframe)
    attacked_fp_dataset = datasets.Dataset(dataframe=attacked_fp_dataset, target_attribute='target', primary_key_attribute='Id')
    suspect = scheme.detection(attacked_fp_dataset, secret_key=4370315727)


def flipping_false_miss_estimation():
    dataset = datasets.GermanCredit()
    parameter_grid = {'fp_len': [32, 64, 128],
                      'gamma': [1, 1.11, 1.25, 1.43, 1.67, 2, 2.5, 3.33, 5, 10],
                      'xi': [1, 2, 4]}
    for fp_len in parameter_grid['fp_len']:
        for gamma in parameter_grid['gamma']:
            for xi in parameter_grid['xi']:
                scheme = Universal(fingerprint_bit_length=fp_len, gamma=gamma, xi=xi)
                false_miss = dict()
                for strength in np.arange(0.0, 1.1, 0.1):
                    attack = attacks.FlippingAttack()
                    false_miss[strength] = attack.false_miss_estimation(dataset=dataset, strength=strength, scheme=scheme)
                with open('robustness/flipping_est/german_credit/false_miss_l{}_g{}_x{}.json'.format(fp_len, gamma, xi),
                          'w') as outfile:
                    json.dump(false_miss, outfile)


def main():
    flipping_false_miss_estimation()


if __name__ == '__main__':
    main()
