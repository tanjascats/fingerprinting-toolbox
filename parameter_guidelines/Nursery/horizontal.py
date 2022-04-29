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
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from scheme import *
from datasets import Nursery
from attacks import HorizontalSubsetAttack


def test_attack():
    scheme = Universal(gamma=1, fingerprint_bit_length=32, xi=1)
    fp_data = pd.read_csv(
        "parameter_guidelines/fingerprinted_data/nursery/attr_subset_8/universal_g1_x1_l32_u1_sk0.csv")
    print(fp_data)
    attack = HorizontalSubsetAttack()
    attacked_data = attack.run(fp_data, strength=0.99, random_state=9)
    print(scheme.detection(attacked_data, secret_key=0, primary_key_attribute='Id', target_attribute='target',
                           original_attributes=['parents', 'has_nurs', 'form', 'children', 'housing', 'finance',
                                                'social', 'health']))


def robustness(scheme, data, exclude=None, include=None, n_experiments=100, confidence_rate=0.99,
               attack_granularity=0.10):
    attack = HorizontalSubsetAttack()
    attack_strength = 1  # defining the strongest attack
    # attack_strength = attack.get_strongest(attack_granularity)  # this should return 0+attack_granularity in case of horizontal subset attack
    # attack_strength = attack.get_weaker(attack_strength, attack_granularity)
    while True:
        # how much data will stay in the release, not how much it will be deleted
        attack_strength -= attack_granularity  # lower the strength of the attack
        if round(attack_strength, 2) <= 0:  # break if the weakest attack is reached
            break
        robust = True  # for now it's robust
        success = n_experiments
        for exp_idx in range(n_experiments):
            # insert the data
            user = 1
            sk = exp_idx
            #fingerprinted_data = scheme.insertion(data, user, secret_key=sk, exclude=exclude,
            #                                      primary_key_attribute=primary_key_attribute)
            if include is None:
                if isinstance(data, Nursery):
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
            attacked_data = attack.run(fingerprinted_data, strength=attack_strength, random_state=sk)

            # try detection
            orig_attr = fingerprinted_data.columns.drop([data.get_target_attribute(),
                                                        data.get_primary_key_attribute()])
            #suspect = scheme.detection(attacked_data, sk, exclude=exclude,
            #                           primary_key_attribute=data.primary_key_attribute(),
            #                           original_attributes=orig_attr)
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
            return round(attack_strength, 2)
    return round(attack_strength, 2)


def robustness_evaluation(confidence_rate, n_experiments, gammae):
    file_string = 'robustness_horizontal_universal_c{}_e{}.pickle'.format(format(confidence_rate, ".2f")[-2:],
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
    xi = 1
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


def horizontal_attack_accuracy_eval(model, X, y, X_fp, y_fp, attack_strength, n_shuffles=5):
    # split the data
    # obtain the baseline acc by training on full train data and evaluating on test
    # attack the TRAIN data, evaluate on the same test set
    baseline_accuracy = []
    attack_accuracy = []
    attack = HorizontalSubsetAttack()
    for fold in range(n_shuffles):
        X_fp_train, X_fp_test, y_fp_train, y_fp_test = train_test_split(X_fp, y_fp, test_size=0.2, random_state=fold, shuffle=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=fold, shuffle=True)

        train_fp = pd.concat([X_fp_train, y_fp_train], axis=1)
        attacked_train = attack.run(train_fp, attack_strength, random_state=fold)
        attacked_X = attacked_train.drop('target', axis=1)  # todo: Id included?
        attacked_y = attacked_train['target']

        model1 = model
        model1.fit(X_fp_train, y_fp_train)
        baseline_acc = accuracy_score(y_test, model1.predict(X_test))
        baseline_accuracy.append(baseline_acc)

        model2 = model
        model2.fit(attacked_X, attacked_y)
        acc = accuracy_score(y_test, model2.predict(X_test))
        attack_accuracy.append(acc)

    return baseline_accuracy, attack_accuracy


def attack_utility(model, gamma, n_attributes):
    # How much utility loss is there if the attacker attacks with the weakest successful attack?

    n_experiments = 10  # number of fingerprinted datasets to consider
    data = Nursery().preprocessed()
    X = data.drop('target', axis=1)
    if 'Id' in X.columns:
        X = X.drop('Id', axis=1)
    y = data['target']


    # 1. define gamma, #attributes marked

    # 2. read the robustness of that parameter setting
    with open('parameter_guidelines/Nursery/evaluation/robustness_horizontal_universal_c95_e30.pickle'
                          , 'rb') as infile:
            robustness_horizontal = pickle.load(infile)
    robustness = robustness_horizontal[gamma]
    print('Robustness: {}'.format(robustness))

    # 3. increase the attack strength by 1 step (i.e. the weakest successful attack)
    step = 0.05
    weakest_successful_attack = round(robustness + step, 2)
    if weakest_successful_attack == 1.0:
        weakest_successful_attack = 0.99
    print('Weakest successful attack: {}'.format(weakest_successful_attack))

    # 4. obtain the baseline utility of the fingerprinted dataset for this parameter setting
    #with open('parameter_guidelines/evaluation/nursery/utility_fp_gb_fpattr8_e30.pickle', 'rb') as infile:
    #    utility_fp_gb = pickle.load(infile)#

    #baseline_utility = utility_fp_gb[gamma]
    #pprint(baseline_utility)  # each value is related to a separate fingerprinted dataset

    # 5. apply the attack
    attack = HorizontalSubsetAttack()
    abs_attack_utility_decrease = []
    rel_attack_utility_decrease = []
    for e in range(n_experiments):  # for each fingerprinted data set
        fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/nursery/'
                                         'attr_subset_{}/universal_g{}_x1_l32_u1_sk{}.csv'.format(
            n_attributes, gamma, e))

        # print(fingerprinted_data.columns)

        n_folds = 5
        fingerprinted_data = Nursery().preprocessed(fingerprinted_data)
        X_fp = fingerprinted_data.drop('target', axis=1)

        if 'Id' in X_fp.columns:
            X_fp = X_fp.drop('Id', axis=1)
        baseline_acc, attacked_acc = horizontal_attack_accuracy_eval(model=model, X=X, y=y,
                                                                     X_fp=X_fp, y_fp=y,
                                                                     attack_strength=weakest_successful_attack,
                                                                     n_shuffles=n_folds)
        print(baseline_acc)
        print(attacked_acc)

        # 7. calculate the difference compared to the baseline
        abs_attack_utility_decrease.append(np.array(attacked_acc) - np.array(baseline_acc))
        rel_attack_utility_decrease.append(
            (np.array(attacked_acc) - np.array(baseline_acc)) / np.array(baseline_acc))

    return abs_attack_utility_decrease, rel_attack_utility_decrease


# attribute model is a model class
def run_utility(model, model_txt):
    gammae = [1, 1.11, 1.25, 1.43, 1.67, 2, 2.5, 3, 4, 5, 10, 18]
    #with open('parameter_guidelines/Nursery/evaluation/abs_horizontal_attack_utility_loss_gb_fpattr8.pickle', 'rb') as infile:
    #   abs_horizontal_utility_loss_fpattr = pickle.load(infile)
    #with open('parameter_guidelines/Nursery/evaluation/rel_horizontal_attack_utility_loss_gb_fpattr8.pickle', 'rb') as infile:
    #   rel_horizontal_utility_loss_fpattr = pickle.load(infile)
    abs_horizontal_utility_loss_fpattr = dict()
    rel_horizontal_utility_loss_fpattr = dict()

    for gamma in gammae:
        abs_attack_utility, rel_attack_utility = attack_utility(model, gamma=gamma, n_attributes=8)  # fold-wise; fingerprinted-data-wise

        abs_horizontal_utility_loss_fpattr[gamma] = abs_attack_utility
        rel_horizontal_utility_loss_fpattr[gamma] = rel_attack_utility

    with open('parameter_guidelines/Nursery/evaluation/abs_horizontal_attack_utility_loss_{}_fpattr8.pickle'.format(model_txt), 'wb') as outfile:
       pickle.dump(abs_horizontal_utility_loss_fpattr, outfile)
    with open('parameter_guidelines/Nursery/evaluation/rel_horizontal_attack_utility_loss_{}_fpattr8.pickle'.format(model_txt), 'wb') as outfile:
       pickle.dump(rel_horizontal_utility_loss_fpattr, outfile)

    pprint(abs_horizontal_utility_loss_fpattr)
    pprint(rel_horizontal_utility_loss_fpattr)


if __name__ == '__main__':
    #test_attack()
    #scheme = Universal(gamma=1, fingerprint_bit_length=32, xi=1)
    #print(robustness(scheme, Nursery(), n_experiments=1, confidence_rate=0.95))
    #gammae = [2, 2.5, 3, 4, 5, 10, 18] # [1, 1.11, 1.25, 1.43, 1.67]
    #robustness_evaluation(confidence_rate=0.95, n_experiments=30, gammae=gammae)
    run_utility(SVC(random_state=0), 'svm')

