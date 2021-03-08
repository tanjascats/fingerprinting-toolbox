import matplotlib.pyplot as plt
import pandas as pd
import os
from pprint import pprint
from utils import *
from scheme import AKScheme
import numpy as np
from attacks import *


def get_experimental_gammae(amount, data_len, fp_len):
    # returns the gammae based on calculation: data_len/(gamma*fp_len) > 1; min gamma is 1
    # gamma values are uniformelly distributed
    # todo: log distribution of gamma values, i.e. get smth like 1, 2, 4, 6, 10, 16, 25...
    min_gamma = 1
    max_gamma = int(data_len/fp_len)
    step = int((max_gamma - min_gamma) / amount)
    gammae = [g for g in range(min_gamma, max_gamma, step)]
    return gammae


def get_marks_per_attribute(path_to_fp_data, original_data):
    percentages = []
    onlyfiles = [f for f in os.listdir(path_to_fp_data) if os.path.isfile(os.path.join(path_to_fp_data, f))]

    for file in onlyfiles:
        fingerprinted_data = pd.read_csv(os.path.join(path_to_fp_data, file))
        percentage = {}
        for index in range(len(original_data.columns)):
            original = original_data[original_data.columns[index]]
            fingerprinted = fingerprinted_data[original_data.columns[index]]
            num_of_changes = len(original.compare(fingerprinted))
            percentage[original_data.columns[index]] = (num_of_changes / len(original_data)) * 100
        percentages.append(percentage)
    return percentages


def get_insights(data, target, primary_key_attribute=None, exclude=None, include=None):
    dataset = None
    if isinstance(data, pd.DataFrame):
        dataset = data
    elif isinstance(data, str):
        print('given the data path')
        dataset = pd.read_csv(data)
    if exclude is None:
        exclude = []
    exclude.append(target)
    fig,ax = plt.subplots()

    # ------------------ #
    # EXPERIMENTAL SETUP #
    # ------------------ #
    fp_len = 8  # for this also figure out a nice setup to choose a good val
    gammae = get_experimental_gammae(3, len(dataset), fp_len)
    xi = 1
    numbuyers = 10

    print('Placeholder for mean / var analysis')
    # define the scheme
    n_experiments = 2
    for exp_idx in range(n_experiments):
        for gamma in gammae:
            # todo: CHANGE SECRET KEY IF OUTER LOOP IS ADDED!
            secret_key = gamma*exp_idx
            scheme = AKScheme(gamma, xi, fp_len, secret_key, numbuyers)
            fingerprinted_data = scheme.insertion(dataset, 1, save=True,
                                                  write_to="parameter_guidelines/evaluation/gamma{}_xi{}_L{}/{}_{}.csv".format(gamma, xi, fp_len, exp_idx, int(time.time())),
                                                  exclude=exclude,
                                                  primary_key_attribute=primary_key_attribute)

    results = {}
    for gamma in gammae:
        marks_percentage_per_attribute = get_marks_per_attribute("parameter_guidelines/evaluation/gamma{}_xi{}_L{}".format(gamma, xi, fp_len), dataset)  # returns a list of 100 evaluated datasets
        pprint(marks_percentage_per_attribute)
        results[gamma] = marks_percentage_per_attribute
    attr = ['clump-thickness']
    print(np.mean(results[1][i]['bare-nuclei'] for i in range(n_experiments)))
    pprint(results)
    print('Placeholder for classification analysis')
    print('Placeholder for robustness analysis via extraction rate')
    print('Placeholder for robustness analysis against attacks')
    pass


# from how much remaining data can the fingerprint still be extracted?
# todo: create a class Dataset that contains these stuff like primary-key-attr, exclude, include and other related stuffs
def inverse_robustness(attack, scheme, data,
                       primary_key_attribute=None, exclude=None, n_experiments=100, confidence_rate=0.99,
                       attack_granularity=0.10):
    attack_strength = 0
    # attack_strength = attack.get_strongest(attack_granularity)  # this should return 0+attack_granularity in case of horizontal subset attack
    # attack_strength = attack.get_weaker(attack_strength, attack_granularity)
    while True:
        attack_strength += attack_granularity  # lower the strength of the attack
        if round(attack_strength, 2) == 1.0:
            break
        robust = True
        success = n_experiments
        for exp_idx in range(n_experiments):
            # insert the data
            user = 1
            sk = exp_idx
            scheme.set_secret_key(sk)
            fingerprinted_data = scheme.insertion(data, user, exclude=exclude,
                                                  primary_key_attribute=primary_key_attribute)
            attacked_data = attack.run(fingerprinted_data, attack_strength)

            # try detection
            suspect = scheme.detection(attacked_data, exclude=exclude, primary_key_attribute=primary_key_attribute)

            if suspect != user:
                success -= 1
            if success / n_experiments < confidence_rate:
                robust = False
                print('-------------------------------------------------------------------')
                print('-------------------------------------------------------------------')
                print(
                    'Attack ' + str(attack_strength) + " is too strong. Halting after " + str(exp_idx) + " iterations.")
                print('-------------------------------------------------------------------')
                print('-------------------------------------------------------------------')
                break  # attack too strong, continue with a lighter one
        if robust:
            return round(attack_strength, 2)
    return round(attack_strength, 2)


def get_robustness(data, primary_key_attribute, target, exclude=None):
    dataset = None
    if isinstance(data, pd.DataFrame):
        dataset = data
    elif isinstance(data, str):
        print('given the data path')
        dataset = pd.read_csv(data)
    if exclude is None:
        exclude = []
    exclude.append(target)

    gammae = [3, 6, 12, 25, 50]
    results = {g: 0 for g in gammae}
    xi = 1
    fplen = 32
    numbuyers = 10
    sk = 123
    attacks = ['horisontal_subset', 'vertical_subset', 'flipping']
    attack = HorizontalSubsetAttack()
    for gamma in gammae:
        scheme = AKScheme(gamma, xi, fplen, sk, numbuyers)
        remaining = inverse_robustness(attack, scheme, dataset, primary_key_attribute=primary_key_attribute,
                                       exclude=[target],
                                       attack_granularity=0.05)
        results[gamma] = remaining
    # todo:plot

    return results
