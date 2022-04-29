from scheme import *
from datasets import Nursery
from attacks import VerticalSubsetAttack
import pandas as pd
import pickle
import os
from pprint import pprint
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from utilities import fp_cross_val_score
import numpy as np
from sklearn.model_selection import cross_val_score


def test():
    scheme = Universal(gamma=1, fingerprint_bit_length=32, xi=1)
    fp_data = pd.read_csv("parameter_guidelines/fingerprinted_data/nursery/attr_subset_8/universal_g1_x1_l32_u1_sk0.csv")
    print(fp_data)
    print(scheme.detection(fp_data, secret_key=0, primary_key_attribute='Id', target_attribute='target',
                           original_attributes=['parents', 'has_nurs', 'form', 'children', 'housing', 'finance',
                                                'social', 'health']))


def test_all():
    scheme = Universal(gamma=1, fingerprint_bit_length=32, xi=1)
    for sk in range(30):
        fp_data = pd.read_csv(
            "parameter_guidelines/fingerprinted_data/nursery/attr_subset_8/"
            "universal_g1_x1_l32_u1_sk{}.csv".format(sk))
        print(scheme.detection(fp_data, secret_key=sk, primary_key_attribute='Id', target_attribute='target',
                               original_attributes=['parents', 'has_nurs', 'form', 'children', 'housing', 'finance',
                                                    'social', 'health']))


def test_attack():
    scheme = Universal(gamma=1, fingerprint_bit_length=32, xi=1)
    fp_data = pd.read_csv(
        "parameter_guidelines/fingerprinted_data/nursery/attr_subset_8/universal_g1_x1_l32_u1_sk0.csv")
    print(fp_data)
    attack = VerticalSubsetAttack()
    attacked_data = attack.run_random(fp_data, 7, seed=0, keep_columns=['Id', 'target'])
    print(scheme.detection(attacked_data, secret_key=0, primary_key_attribute='Id', target_attribute='target',
                           original_attributes=['parents', 'has_nurs', 'form', 'children', 'housing', 'finance',
                                                'social', 'health']))


# returns robustness (=from how much remaining data can the fingerprint still be extracted with confidence_rate based
# on n_experiments)
# attack and scheme need to be provided as instances of Attack and Scheme abstract classes, respectively
def robustness(scheme, data, exclude=None, include=None, n_experiments=30, confidence_rate=0.99):
    attack = VerticalSubsetAttack()
    attack_strength = 1  # defining strongest attack
    attack_vertical_max = -1
    while True:
        attack_strength -= 1  # lower the strength of the attack
        if attack_strength == 0 and attack_vertical_max != -1:
            break

        robust = True  # for now it's robust
        success = n_experiments
        for exp_idx in range(n_experiments):
            # insert the data
            user = 1
            sk = exp_idx
            if include is None:
                fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/' + data.to_string() +
                                                 '/attr_subset_8' +
                                                 '/universal_g{}_x{}_l{}_u{}_sk{}.csv'.format(scheme.get_gamma(), 1,
                                                                                                   scheme.get_fplen(),
                                                                                                   user, sk))
            else:
                fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/' + data.to_string() +
                                                 '/attr_subset_' + str(len(include)) +
                                                 '/universal_g{}_x{}_l{}_u{}_sk{}.csv'.format(scheme.get_gamma(), 1,
                                                                                              scheme.get_fplen(),
                                                                                              user, sk))
            if attack_vertical_max == -1:  # remember the strongest attack and initiate the attack strength
                attack_vertical_max = len(fingerprinted_data.columns.drop([data.get_target_attribute(),
                                                                           data.get_primary_key_attribute()]))
                attack_strength = attack_vertical_max - 1
            attacked_data = attack.run_random(fingerprinted_data, attack_strength,
                                              keep_columns=[data.get_target_attribute(),
                                                            data.get_primary_key_attribute()], seed=sk)
             # try detection
            if include is not None:
                original_attributes = pd.Series(data=include)
            else:
                original_attributes = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance',
                                                'social', 'health']
            suspect = scheme.detection(attacked_data, secret_key=sk, primary_key_attribute='Id',
                                       original_attributes=original_attributes,
                                       target_attribute='target')

            if suspect != user:
                success -= 1
            if success / n_experiments < confidence_rate:
                robust = False
                print('-------------------------------------------------------------------')
                print('-------------------------------------------------------------------')
                print(
                    'Attack with strength ' + str(attack_strength) + " is too strong. Halting after " + str(exp_idx) +
                    " iterations.")
                print('-------------------------------------------------------------------')
                print('-------------------------------------------------------------------')
                break  # attack too strong, continue with a lighter one
        if robust:
            if isinstance(attack, VerticalSubsetAttack):
                attack_strength = round(attack_strength / attack_vertical_max, 2)
            return round(attack_strength, 2)
    if isinstance(attack, VerticalSubsetAttack):
        attack_strength = round(attack_strength / attack_vertical_max, 2)
    # todo: mirror the performance for >0.5 flipping attacks
    return round(attack_strength, 2)


def robustness_evaluation(confidence_rate, n_experiments, gammae):
    file_string = 'robustness_vertical_universal_c{}_e{}.pickle'.format(format(confidence_rate, ".2f")[-2:],
                                                                  n_experiments)
    # check if results exist
    # ---------------------- #
    if os.path.isfile('parameter_guidelines/Nursery/evaluation/' + file_string):
        with open('parameter_guidelines/Nursery/evaluation/' + file_string, 'rb') as infile:
            resutls = pickle.load(infile)
    else:
        resutls = {}
    gammae_new = []
    for gamma in gammae:
        if gamma not in resutls.keys():
            gammae_new.append(gamma)
            print('Updating results with gamma={}'.format(gamma))
    # ---------------------- #
    xi=1
    fplen=32
    for gamma in gammae_new:
        scheme = Universal(gamma=gamma,
                           xi=xi,
                           fingerprint_bit_length=fplen)
        # from how much remaining data can the fingerprint still be extracted?
        remaining = robustness(scheme, Nursery(),
                               n_experiments=n_experiments,
                               confidence_rate=confidence_rate)
        resutls[gamma] = remaining
    resutls = dict(sorted(resutls.items()))
    # todo: remove comment
    print(resutls)
    with open('parameter_guidelines/Nursery/evaluation/' + file_string, 'wb') as outfile:
        pickle.dump(resutls, outfile)


def attack_utility_gb(gamma, n_attributes):
    # How much utility loss is there if the attacker attacks with the weakest successful attack?

    n_experiments = 10  # number of fingerprinted datasets to consider
    data = Nursery().preprocessed()
    X = data.drop('target', axis=1)
    if 'Id' in X.columns:
        X = X.drop('Id', axis=1)
    y = data['target']

    # 1. define gamma, #attributes marked

    # 2. read the robustness of that parameter setting
    with open('parameter_guidelines/Nursery/evaluation/robustness_vertical_universal_c95_e30.pickle'
                          , 'rb') as infile:
            robustness_vertical = pickle.load(infile)
    robustness = robustness_vertical[gamma]
    print('Robustness: {}'.format(robustness))

    # 3. increase the attack strength by 1 step (i.e. the weakest successful attack)
    robustness_absolute = robustness*n_attributes
    step = 1
    weakest_successful_attack = int(robustness_absolute+step)
    print('Weakest successful attack: {}'.format(weakest_successful_attack))

    # 4. obtain the baseline utility of the fingerprinted dataset for this parameter setting
    with open('parameter_guidelines/evaluation/nursery/utility_fp_gb_fpattr8_e30.pickle', 'rb') as infile:
        utility_fp_gb = pickle.load(infile)

    baseline_utility = utility_fp_gb[gamma]
    pprint(baseline_utility)  # each value is related to a separate fingerprinted dataset

    # 5. apply the attack
    attack = VerticalSubsetAttack()
    abs_attack_utility_decrease = []
    rel_attack_utility_decrease = []
    for e in range(n_experiments):  # for each fingerprinted data set
        fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/nursery/attr_subset_8/'
                                         'universal_g{}_x1_l32_u1_sk{}.csv'.format(gamma, e))
        print(fingerprinted_data.columns)
        attacked_data = attack.run_random(fingerprinted_data, weakest_successful_attack, seed=e,
                                          keep_columns=['target'])
        print(attacked_data.columns)

    # 6. evaluate the performance of the attacked data
        model = GradientBoostingClassifier(random_state=0)
        attacked_data = Nursery().preprocessed(attacked_data)
        X_attacked = attacked_data.drop(['target'], axis=1)
        y_attacked = attacked_data['target']
        n_folds = 3

        X_original = X[X_attacked.columns]
        y_original = y
        attacked_acc = fp_cross_val_score(model, X_original, y_original, X_attacked, y_attacked, cv=n_folds, scoring='accuracy')['test_score']
        print('accuracy diff: {} vs {}'.format(baseline_utility[e], attacked_acc))
        # 7. calculate the difference compared to the baseline
        print(np.array(attacked_acc) - np.array(baseline_utility[e]))
        abs_attack_utility_decrease.append(np.array(attacked_acc) - np.array(baseline_utility[e]))
        rel_attack_utility_decrease.append((np.array(attacked_acc) - np.array(baseline_utility[e])) / np.array(baseline_utility[e]))

    return abs_attack_utility_decrease, rel_attack_utility_decrease


def attack_utility_mlp(gamma, n_attributes):
    # How much utility loss is there if the attacker attacks with the weakest successful attack?

    n_experiments = 10  # number of fingerprinted datasets to consider
    data = Nursery().preprocessed()
    X = data.drop('target', axis=1)
    if 'Id' in X.columns:
        X = X.drop('Id', axis=1)
    y = data['target']

    # 1. define gamma, #attributes marked

    # 2. read the robustness of that parameter setting
    with open('parameter_guidelines/Nursery/evaluation/robustness_vertical_universal_c95_e30.pickle'
                          , 'rb') as infile:
            robustness_vertical = pickle.load(infile)
    robustness = robustness_vertical[gamma]
    print('Robustness: {}'.format(robustness))

    # 3. increase the attack strength by 1 step (i.e. the weakest successful attack)
    robustness_absolute = robustness*n_attributes
    step = 1
    weakest_successful_attack = int(robustness_absolute+step)
    print('Weakest successful attack: {}'.format(weakest_successful_attack))

    # 4. obtain the baseline utility of the fingerprinted dataset for this parameter setting
    with open('parameter_guidelines/evaluation/nursery/utility_fp_mlp_fpattr8_e30.pickle', 'rb') as infile:
        utility_fp_gb = pickle.load(infile)

    baseline_utility = utility_fp_gb[gamma]
    pprint(baseline_utility)  # each value is related to a separate fingerprinted dataset

    # 5. apply the attack
    attack = VerticalSubsetAttack()
    abs_attack_utility_decrease = []
    rel_attack_utility_decrease = []
    for e in range(n_experiments):  # for each fingerprinted data set
        fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/nursery/attr_subset_8/'
                                         'universal_g{}_x1_l32_u1_sk{}.csv'.format(gamma, e))
        print(fingerprinted_data.columns)
        attacked_data = attack.run_random(fingerprinted_data, weakest_successful_attack, seed=e,
                                          keep_columns=['target'])
        print(attacked_data.columns)

    # 6. evaluate the performance of the attacked data
        model = MLPClassifier(random_state=0)
        attacked_data = Nursery().preprocessed(attacked_data)
        X_attacked = attacked_data.drop(['target'], axis=1)
        y_attacked = attacked_data['target']
        n_folds = 3

        X_original = X[X_attacked.columns]
        y_original = y
        attacked_acc = fp_cross_val_score(model, X_original, y_original, X_attacked, y_attacked, cv=n_folds, scoring='accuracy')['test_score']
        print('accuracy diff: {} vs {}'.format(baseline_utility[e], attacked_acc))
        # 7. calculate the difference compared to the baseline
        print(np.array(attacked_acc) - np.array(baseline_utility[e]))
        abs_attack_utility_decrease.append(np.array(attacked_acc) - np.array(baseline_utility[e]))
        rel_attack_utility_decrease.append((np.array(attacked_acc) - np.array(baseline_utility[e])) / np.array(baseline_utility[e]))

    return abs_attack_utility_decrease, rel_attack_utility_decrease


def attack_utility_knn(gamma, n_attributes):
    # How much utility loss is there if the attacker attacks with the weakest successful attack?

    n_experiments = 10  # number of fingerprinted datasets to consider
    data = Nursery().preprocessed()
    X = data.drop('target', axis=1)
    if 'Id' in X.columns:
        X = X.drop('Id', axis=1)
    y = data['target']

    # 1. define gamma, #attributes marked

    # 2. read the robustness of that parameter setting
    with open('parameter_guidelines/Nursery/evaluation/robustness_vertical_universal_c95_e30.pickle'
            , 'rb') as infile:
        robustness_vertical = pickle.load(infile)
    robustness = robustness_vertical[gamma]
    print('Robustness: {}'.format(robustness))

    # 3. increase the attack strength by 1 step (i.e. the weakest successful attack)
    robustness_absolute = robustness * n_attributes
    step = 1
    weakest_successful_attack = int(robustness_absolute + step)
    print('Weakest successful attack: {}'.format(weakest_successful_attack))

    # 4. obtain the baseline utility of the fingerprinted dataset for this parameter setting
    with open('parameter_guidelines/evaluation/nursery/utility_fp_knn_fpattr8_e30.pickle', 'rb') as infile:
        utility_fp_gb = pickle.load(infile)

    baseline_utility = utility_fp_gb[gamma]
    pprint(baseline_utility)  # each value is related to a separate fingerprinted dataset

    # 5. apply the attack
    attack = VerticalSubsetAttack()
    abs_attack_utility_decrease = []
    rel_attack_utility_decrease = []
    for e in range(n_experiments):  # for each fingerprinted data set
        fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/nursery/attr_subset_8/'
                                         'universal_g{}_x1_l32_u1_sk{}.csv'.format(gamma, e))
        print(fingerprinted_data.columns)
        attacked_data = attack.run_random(fingerprinted_data, weakest_successful_attack, seed=e,
                                          keep_columns=['target'])
        print(attacked_data.columns)

        # 6. evaluate the performance of the attacked data
        model = KNeighborsClassifier()
        attacked_data = Nursery().preprocessed(attacked_data)
        X_attacked = attacked_data.drop(['target'], axis=1)
        y_attacked = attacked_data['target']
        n_folds = 3

        X_original = X[X_attacked.columns]
        y_original = y
        attacked_acc = \
        fp_cross_val_score(model, X_original, y_original, X_attacked, y_attacked, cv=n_folds, scoring='accuracy')[
            'test_score']
        print('accuracy diff: {} vs {}'.format(baseline_utility[e], attacked_acc))
        # 7. calculate the difference compared to the baseline
        print(np.array(attacked_acc) - np.array(baseline_utility[e]))
        abs_attack_utility_decrease.append(np.array(attacked_acc) - np.array(baseline_utility[e]))
        rel_attack_utility_decrease.append(
            (np.array(attacked_acc) - np.array(baseline_utility[e])) / np.array(baseline_utility[e]))

    return abs_attack_utility_decrease, rel_attack_utility_decrease


def attack_utility_lr(gamma, n_attributes):
    # How much utility loss is there if the attacker attacks with the weakest successful attack?

    n_experiments = 10  # number of fingerprinted datasets to consider
    data = Nursery().preprocessed()
    X = data.drop('target', axis=1)
    if 'Id' in X.columns:
        X = X.drop('Id', axis=1)
    y = data['target']

    # 1. define gamma, #attributes marked

    # 2. read the robustness of that parameter setting
    with open('parameter_guidelines/Nursery/evaluation/robustness_vertical_universal_c95_e30.pickle'
            , 'rb') as infile:
        robustness_vertical = pickle.load(infile)
    robustness = robustness_vertical[gamma]
    print('Robustness: {}'.format(robustness))

    # 3. increase the attack strength by 1 step (i.e. the weakest successful attack)
    robustness_absolute = robustness * n_attributes
    step = 1
    weakest_successful_attack = int(robustness_absolute + step)
    print('Weakest successful attack: {}'.format(weakest_successful_attack))

    # 4. obtain the baseline utility of the fingerprinted dataset for this parameter setting
    with open('parameter_guidelines/evaluation/nursery/utility_fp_lr_fpattr8_e30.pickle', 'rb') as infile:
        utility_fp_gb = pickle.load(infile)

    baseline_utility = utility_fp_gb[gamma]
    pprint(baseline_utility)  # each value is related to a separate fingerprinted dataset

    # 5. apply the attack
    attack = VerticalSubsetAttack()
    abs_attack_utility_decrease = []
    rel_attack_utility_decrease = []
    for e in range(n_experiments):  # for each fingerprinted data set
        fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/nursery/attr_subset_8/'
                                         'universal_g{}_x1_l32_u1_sk{}.csv'.format(gamma, e))
        print(fingerprinted_data.columns)
        attacked_data = attack.run_random(fingerprinted_data, weakest_successful_attack, seed=e,
                                          keep_columns=['target'])
        print(attacked_data.columns)

        # 6. evaluate the performance of the attacked data
        model = LogisticRegression(random_state=0)
        attacked_data = Nursery().preprocessed(attacked_data)
        X_attacked = attacked_data.drop(['target'], axis=1)
        y_attacked = attacked_data['target']
        n_folds = 3

        X_original = X[X_attacked.columns]
        y_original = y
        attacked_acc = \
        fp_cross_val_score(model, X_original, y_original, X_attacked, y_attacked, cv=n_folds, scoring='accuracy')[
            'test_score']
        print('accuracy diff: {} vs {}'.format(baseline_utility[e], attacked_acc))
        # 7. calculate the difference compared to the baseline
        print(np.array(attacked_acc) - np.array(baseline_utility[e]))
        abs_attack_utility_decrease.append(np.array(attacked_acc) - np.array(baseline_utility[e]))
        rel_attack_utility_decrease.append(
            (np.array(attacked_acc) - np.array(baseline_utility[e])) / np.array(baseline_utility[e]))

    return abs_attack_utility_decrease, rel_attack_utility_decrease


def attack_utility_svm(gamma, n_attributes):
    # How much utility loss is there if the attacker attacks with the weakest successful attack?

    n_experiments = 10  # number of fingerprinted datasets to consider
    data = Nursery().preprocessed()
    X = data.drop('target', axis=1)
    if 'Id' in X.columns:
        X = X.drop('Id', axis=1)
    y = data['target']

    # 1. define gamma, #attributes marked

    # 2. read the robustness of that parameter setting
    with open('parameter_guidelines/Nursery/evaluation/robustness_vertical_universal_c95_e30.pickle'
            , 'rb') as infile:
        robustness_vertical = pickle.load(infile)
    robustness = robustness_vertical[gamma]
    print('Robustness: {}'.format(robustness))

    # 3. increase the attack strength by 1 step (i.e. the weakest successful attack)
    robustness_absolute = robustness * n_attributes
    step = 1
    weakest_successful_attack = int(robustness_absolute + step)
    print('Weakest successful attack: {}'.format(weakest_successful_attack))

    # 4. obtain the baseline utility of the fingerprinted dataset for this parameter setting
    with open('parameter_guidelines/evaluation/nursery/utility_fp_svm_fpattr8_e30.pickle', 'rb') as infile:
        utility_fp_gb = pickle.load(infile)

    baseline_utility = utility_fp_gb[gamma]
    pprint(baseline_utility)  # each value is related to a separate fingerprinted dataset

    # 5. apply the attack
    attack = VerticalSubsetAttack()
    abs_attack_utility_decrease = []
    rel_attack_utility_decrease = []
    for e in range(n_experiments):  # for each fingerprinted data set
        fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/nursery/attr_subset_8/'
                                         'universal_g{}_x1_l32_u1_sk{}.csv'.format(gamma, e))
        print(fingerprinted_data.columns)
        attacked_data = attack.run_random(fingerprinted_data, weakest_successful_attack, seed=e,
                                          keep_columns=['target'])
        print(attacked_data.columns)

        # 6. evaluate the performance of the attacked data
        model = SVC(random_state=0)
        attacked_data = Nursery().preprocessed(attacked_data)
        X_attacked = attacked_data.drop(['target'], axis=1)
        y_attacked = attacked_data['target']
        n_folds = 3

        X_original = X[X_attacked.columns]
        y_original = y
        attacked_acc = \
        fp_cross_val_score(model, X_original, y_original, X_attacked, y_attacked, cv=n_folds, scoring='accuracy')[
            'test_score']
        print('accuracy diff: {} vs {}'.format(baseline_utility[e], attacked_acc))
        # 7. calculate the difference compared to the baseline
        print(np.array(attacked_acc) - np.array(baseline_utility[e]))
        abs_attack_utility_decrease.append(np.array(attacked_acc) - np.array(baseline_utility[e]))
        rel_attack_utility_decrease.append(
            (np.array(attacked_acc) - np.array(baseline_utility[e])) / np.array(baseline_utility[e]))

    return abs_attack_utility_decrease, rel_attack_utility_decrease


def attack_utility_rf(gamma, n_attributes):
    # How much utility loss is there if the attacker attacks with the weakest successful attack?

    n_experiments = 10  # number of fingerprinted datasets to consider
    data = Nursery().preprocessed()
    X = data.drop('target', axis=1)
    if 'Id' in X.columns:
        X = X.drop('Id', axis=1)
    y = data['target']

    # 1. define gamma, #attributes marked

    # 2. read the robustness of that parameter setting
    with open('parameter_guidelines/Nursery/evaluation/robustness_vertical_universal_c95_e30.pickle'
            , 'rb') as infile:
        robustness_vertical = pickle.load(infile)
    robustness = robustness_vertical[gamma]
    print('Robustness: {}'.format(robustness))

    # 3. increase the attack strength by 1 step (i.e. the weakest successful attack)
    robustness_absolute = robustness * n_attributes
    step = 1
    weakest_successful_attack = int(robustness_absolute + step)
    print('Weakest successful attack: {}'.format(weakest_successful_attack))

    # 4. obtain the baseline utility of the fingerprinted dataset for this parameter setting
    with open('parameter_guidelines/evaluation/nursery/utility_fp_rf_fpattr8_e30.pickle', 'rb') as infile:
        utility_fp_gb = pickle.load(infile)

    baseline_utility = utility_fp_gb[gamma]
    pprint(baseline_utility)  # each value is related to a separate fingerprinted dataset

    # 5. apply the attack
    attack = VerticalSubsetAttack()
    abs_attack_utility_decrease = []
    rel_attack_utility_decrease = []
    for e in range(n_experiments):  # for each fingerprinted data set
        fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/nursery/attr_subset_8/'
                                         'universal_g{}_x1_l32_u1_sk{}.csv'.format(gamma, e))
        print(fingerprinted_data.columns)
        attacked_data = attack.run_random(fingerprinted_data, weakest_successful_attack, seed=e,
                                          keep_columns=['target'])
        print(attacked_data.columns)

        # 6. evaluate the performance of the attacked data
        model = RandomForestClassifier(random_state=0)
        attacked_data = Nursery().preprocessed(attacked_data)
        X_attacked = attacked_data.drop(['target'], axis=1)
        y_attacked = attacked_data['target']
        n_folds = 3

        X_original = X[X_attacked.columns]
        y_original = y
        attacked_acc = \
        fp_cross_val_score(model, X_original, y_original, X_attacked, y_attacked, cv=n_folds, scoring='accuracy')[
            'test_score']
        print('accuracy diff: {} vs {}'.format(baseline_utility[e], attacked_acc))
        # 7. calculate the difference compared to the baseline
        print(np.array(attacked_acc) - np.array(baseline_utility[e]))
        abs_attack_utility_decrease.append(np.array(attacked_acc) - np.array(baseline_utility[e]))
        rel_attack_utility_decrease.append(
            (np.array(attacked_acc) - np.array(baseline_utility[e])) / np.array(baseline_utility[e]))

    return abs_attack_utility_decrease, rel_attack_utility_decrease


def run_utility(model):
    gammae = [1, 1.11, 1.25, 1.43, 1.67, 2, 2.5, 3, 4, 5, 10, 18]
    #with open('parameter_guidelines/Nursery/evaluation/abs_vertical_attack_utility_loss_gb_fpattr8.pickle', 'rb') as infile:
    #   abs_vertical_utility_loss_fpattr = pickle.load(infile)
    #with open('parameter_guidelines/Nursery/evaluation/rel_vertical_attack_utility_loss_gb_fpattr8.pickle', 'rb') as infile:
    #   rel_vertical_utility_loss_fpattr = pickle.load(infile)
    abs_vertical_utility_loss_fpattr = dict()
    rel_vertical_utility_loss_fpattr = dict()

    for gamma in gammae:
        if model == 'gb':
            abs_attack_utility, rel_attack_utility = attack_utility_gb(gamma=gamma, n_attributes=8)  # fold-wise; fingerprinted-data-wise
        elif model == 'mlp':
            abs_attack_utility, rel_attack_utility = attack_utility_mlp(gamma=gamma, n_attributes=8)
        elif model == 'knn':
            abs_attack_utility, rel_attack_utility = attack_utility_knn(gamma=gamma, n_attributes=8)
        elif model == 'lr':
            abs_attack_utility, rel_attack_utility = attack_utility_lr(gamma=gamma, n_attributes=8)
        elif model=='svm':
            abs_attack_utility, rel_attack_utility = attack_utility_svm(gamma=gamma, n_attributes=8)
        elif model=='rf':
            abs_attack_utility, rel_attack_utility = attack_utility_rf(gamma=gamma, n_attributes=8)
        abs_vertical_utility_loss_fpattr[gamma] = abs_attack_utility
        rel_vertical_utility_loss_fpattr[gamma] = rel_attack_utility

    with open('parameter_guidelines/Nursery/evaluation/abs_vertical_attack_utility_loss_{}_fpattr8.pickle'.format(model), 'wb') as outfile:
       pickle.dump(abs_vertical_utility_loss_fpattr, outfile)
    with open('parameter_guidelines/Nursery/evaluation/rel_vertical_attack_utility_loss_{}_fpattr8.pickle'.format(model), 'wb') as outfile:
       pickle.dump(rel_vertical_utility_loss_fpattr, outfile)

    pprint(abs_vertical_utility_loss_fpattr)
    pprint(rel_vertical_utility_loss_fpattr)


if __name__ == '__main__':
    #gammae = [2, 2.5, 3, 4, 5, 10, 18] # [1, 1.11, 1.25, 1.43, 1.67]
    #robustness_evaluation(0.95, 30, gammae)
    #print(attack_utility(gamma=18, n_attributes=8))
    #test_fp_cross_val(1, 0)
    run_utility('knn')