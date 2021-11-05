import pickle
from pprint import pprint
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utilities import *
from attacks import *
from datasets import *


def vertical_attack(gamma, n_attributes):
    # How much utility loss is there if the attacker attacks with the weakest successful attack?

    n_experiments = 10  # number of fingerprinted datasets to consider
    data = GermanCredit().preprocessed()
    X = data.drop('target', axis=1)
    if 'Id' in X.columns:
        X = X.drop('Id', axis=1)
    y = data['target']

    # 1. define gamma, #attributes marked

    # 2. read the robustness of that parameter setting
    if n_attributes == 20:
        with open('parameter_guidelines/evaluation/german_credit/robustness_vertical_universal_c95_e100.pickle',
                  'rb') as infile:
            robustness_vertical_nattr = pickle.load(infile)
    else:
        with open('parameter_guidelines/evaluation/german_credit/robustness_vertical_universal_c95_fpattr{}_e100.pickle'
                          .format(n_attributes), 'rb') as infile:
            robustness_vertical_nattr = pickle.load(infile)
    robustness = robustness_vertical_nattr[gamma]
    #print('Robustness: {}'.format(robustness))

    # 3. increase the attack strength by 1 step (i.e. the weakest successful attack)
    robustness_absolute = robustness*n_attributes
    step = 1
    weakest_successful_attack = int(robustness_absolute+step)
    #print('Weakest successful attack: {}'.format(weakest_successful_attack))

    # 4. obtain the baseline utility of the fingerprinted dataset for this parameter setting
    if n_attributes == 20:
        with open('parameter_guidelines/evaluation/german_credit/utility_fp_gb_e80.pickle', 'rb') as infile:
            utility_fp_gb_e80 = pickle.load(infile)
    else:
        with open('parameter_guidelines/evaluation/german_credit/utility_fp_gb_fpattr{}_e80.pickle'.format(
                n_attributes), 'rb') as infile:
            utility_fp_gb_e80 = pickle.load(infile)
    baseline_utility = utility_fp_gb_e80[gamma]
    pprint(baseline_utility)  # each value is related to a separate fingerprinted dataset

    # 5. apply the attack
    attack = VerticalSubsetAttack()
    abs_attack_utility_decrease = []
    rel_attack_utility_decrease = []
    for e in range(n_experiments):  # for each fingerprinted data set
        fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/german_credit/attr_subset_{}/'
                                         'universal_g{}_x1_l8_u1_sk{}.csv'.format(n_attributes, gamma, e))
        #print(fingerprinted_data.columns)
        attacked_data = attack.run_random(fingerprinted_data, weakest_successful_attack, seed=e,
                                          keep_columns=['target'])
        #print(attacked_data.columns)

    # 6. evaluate the performance of the attacked data
        model = GradientBoostingClassifier(random_state=0)
        attacked_data = GermanCredit().preprocessed(attacked_data)
        X_attacked = attacked_data.drop(['target'], axis=1)
        y_attacked = attacked_data['target']
        n_folds = 5

        X_original = X[X_attacked.columns]
        y_original = y
        attacked_acc = fp_cross_val_score(model, X_original, y_original, X_attacked, y_attacked, cv=n_folds, scoring='accuracy')['test_score']
        #print('accuracy diff: {} vs {}'.format(baseline_utility[e], attacked_acc))
        # 7. calculate the difference compared to the baseline
        #print(np.array(attacked_acc) - np.array(baseline_utility[e]))
        abs_attack_utility_decrease.append(np.array(attacked_acc) - np.array(baseline_utility[e]))
        rel_attack_utility_decrease.append((np.array(attacked_acc) - np.array(baseline_utility[e])) / np.array(baseline_utility[e]))

    return abs_attack_utility_decrease, rel_attack_utility_decrease


def flipping_attack(gamma, n_attributes):
    # how much utility loss is there if the attacker attacks with the weakest successful attack?

    n_experiments = 10  # number of fingerprinted datasets to consider
    data = GermanCredit().preprocessed()
    X = data.drop('target', axis=1)
    if 'Id' in X.columns:
        X = X.drop('Id', axis=1)
    y = data['target']

    # 1. define gamma, #attributes marked
    # 2. read the robustness of that parameter setting

    with open('parameter_guidelines/evaluation/german_credit/robustness_flipping_universal_c95_ag05_fpattr{}_e100.pickle'.format(n_attributes), 'rb') as infile:
        robustness_flipping_nattr = pickle.load(infile)
    print(robustness_flipping_nattr)
    robustness = robustness_flipping_nattr[gamma]
    print('Robustness: {}'.format(robustness))

    # 3. increase the attack strength by 1 step (i.e. the weakest successful attack)
    step = 0.05
    weakest_successful_attack = round(robustness + step,2)
    print('Weakest successful attack: {}'.format(weakest_successful_attack))

    # 4. obtain the baseline utility of the fingerprinted dataset for this parameter setting
    if n_attributes == 20:
        with open('parameter_guidelines/evaluation/german_credit/utility_fp_gb_e80.pickle', 'rb') as infile:
            utility_fp_gb_e80 = pickle.load(infile)
    else:
        with open('parameter_guidelines/evaluation/german_credit/utility_fp_gb_fpattr{}_e80.pickle'.format(
                n_attributes), 'rb') as infile:
            utility_fp_gb_e80 = pickle.load(infile)
    baseline_utility = utility_fp_gb_e80[gamma]
    pprint(baseline_utility)  # each value is related to a separate fingerprinted dataset

    # 5. apply the attack
    attack = FlippingAttack()
    abs_attack_utility_decrease = []
    rel_attack_utility_decrease = []
    for e in range(n_experiments):  # for each fingerprinted data set
        fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/german_credit/'
                                         'attr_subset_{}/universal_g{}_x1_l8_u1_sk{}.csv'.format(
            n_attributes, gamma, e))
        # print(fingerprinted_data.columns)
        attacked_data = attack.run(fingerprinted_data, weakest_successful_attack, random_state=e,
                                   keep_columns=['target'])

    # 6. evaluate the performance of the attacked data
        model = GradientBoostingClassifier(random_state=0)
        attacked_data = GermanCredit().preprocessed(attacked_data)
        X_attacked = attacked_data.drop(['target'], axis=1)
        y_attacked = attacked_data['target']
        n_folds = 5

        X_original = X[X_attacked.columns]
        y_original = y
        attacked_acc = \
        fp_cross_val_score(model, X_original, y_original, X_attacked, y_attacked, cv=n_folds, scoring='accuracy')[
            'test_score']
        # print('accuracy diff: {} vs {}'.format(baseline_utility[e], attacked_acc))

    # 7. calculate the difference compared to the baseline
        # print(np.array(attacked_acc) - np.array(baseline_utility[e]))
        abs_attack_utility_decrease.append(np.array(attacked_acc) - np.array(baseline_utility[e]))
        rel_attack_utility_decrease.append(
            (np.array(attacked_acc) - np.array(baseline_utility[e])) / np.array(baseline_utility[e]))

    return abs_attack_utility_decrease, rel_attack_utility_decrease


def run_flipping(n_attr):
    gammae = [1, 1.11, 1.25, 1.43, 1.67, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18]
    if os.path.isfile('parameter_guidelines/evaluation/german_credit/abs_flipping_attack_utility_loss_gb_fpattr{}.pickle'.format(n_attr)):
        with open(
                'parameter_guidelines/evaluation/german_credit/abs_flipping_attack_utility_loss_gb_fpattr{}.pickle'.format(n_attr), 'rb') as infile:
            abs_flipping_utility_loss_fpattr = pickle.load(infile)
    else:
        abs_flipping_utility_loss_fpattr = dict()
    if os.path.isfile(
            'parameter_guidelines/evaluation/german_credit/rel_flipping_attack_utility_loss_gb_fpattr{}.pickle'.format(n_attr)):
        with open(
                'parameter_guidelines/evaluation/german_credit/rel_flipping_attack_utility_loss_gb_fpattr{}.pickle'.format(n_attr), 'rb') as infile:
            rel_flipping_utility_loss_fpattr = pickle.load(infile)
    else:
        rel_flipping_utility_loss_fpattr = dict()

    for gamma in gammae:
        abs_attack_utility, rel_attack_utility = flipping_attack(gamma=gamma, n_attributes=n_attr)  # fold-wise; fingerprinted-data-wise
        abs_flipping_utility_loss_fpattr[gamma] = abs_attack_utility
        rel_flipping_utility_loss_fpattr[gamma] = rel_attack_utility

    with open('parameter_guidelines/evaluation/german_credit/abs_flipping_attack_utility_loss_gb_fpattr{}.pickle'.format(n_attr), 'wb') as outfile:
        pickle.dump(abs_flipping_utility_loss_fpattr, outfile)
    with open('parameter_guidelines/evaluation/german_credit/rel_flipping_attack_utility_loss_gb_fpattr{}.pickle'.format(n_attr), 'wb') as outfile:
        pickle.dump(rel_flipping_utility_loss_fpattr, outfile)

    pprint(abs_flipping_utility_loss_fpattr)
    pprint(rel_flipping_utility_loss_fpattr)


def horizontal_attack_accuracy_eval(X, y, X_fp, y_fp, attack_strength, n_shuffles=5):
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

        model = GradientBoostingClassifier(random_state=0)
        model.fit(X_fp_train, y_fp_train)
        baseline_acc = accuracy_score(y_test, model.predict(X_test))
        baseline_accuracy.append(baseline_acc)

        model = GradientBoostingClassifier(random_state=0)
        model.fit(attacked_X, attacked_y)
        acc = accuracy_score(y_test, model.predict(X_test))
        attack_accuracy.append(acc)

    return baseline_accuracy, attack_accuracy


def horizontal_attack(gamma, n_attributes):
    # how much utility loss is there if the attacker attacks with the weakest successful attack?
    n_experiments = 10  # number of fingerprinted datasets to consider
    data = GermanCredit().preprocessed()
    X = data.drop('target', axis=1)
    if 'Id' in X.columns:
        X = X.drop('Id', axis=1)
    y = data['target']

    # 1. define gamma, #attributes marked
    # 2. read the robustness of that parameter setting
    with open('parameter_guidelines/evaluation/german_credit/robustness_horizontal_universal_c95_ag05_fpattr{}_'
              'e100.pickle'.format(n_attributes), 'rb') as infile:
        robustness_flipping_nattr = pickle.load(infile)
    robustness = robustness_flipping_nattr[gamma]
    print('Robustness: {}'.format(robustness))

    # 3. increase the attack strength by 1 step (i.e. the weakest successful attack)
    step = 0.05
    weakest_successful_attack = round(robustness + step, 2)
    if weakest_successful_attack == 1.0:
        weakest_successful_attack = 0.99
    print('Weakest successful attack: {}'.format(weakest_successful_attack))

    # 5. apply the attack and evaluate
    attack = HorizontalSubsetAttack()
    abs_attack_utility_decrease = []
    rel_attack_utility_decrease = []
    for e in range(n_experiments):  # for each fingerprinted data set
        fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/german_credit/'
                                         'attr_subset_{}/universal_g{}_x1_l8_u1_sk{}.csv'.format(
            n_attributes, gamma, e))
        # print(fingerprinted_data.columns)

        n_folds = 5
        fingerprinted_data = GermanCredit().preprocessed(fingerprinted_data)
        X_fp = fingerprinted_data.drop('target', axis=1)
        if 'Id' in X_fp.columns:
            X_fp = X_fp.drop('Id', axis=1)
        baseline_acc, attacked_acc = horizontal_attack_accuracy_eval(X=X, y=y,
                                                       X_fp=X_fp, y_fp=y,
                                                       attack_strength=weakest_successful_attack, n_shuffles=n_folds)
        print(baseline_acc)
        print(attacked_acc)

        # 7. calculate the difference compared to the baseline
        abs_attack_utility_decrease.append(np.array(attacked_acc) - np.array(baseline_acc))
        rel_attack_utility_decrease.append(
            (np.array(attacked_acc) - np.array(baseline_acc)) / np.array(baseline_acc))

    return abs_attack_utility_decrease, rel_attack_utility_decrease


def run_horizontal(n_attr):
    gammae = [1, 1.11, 1.25, 1.43, 1.67, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18]

    #gammae = [2.5, 1.67, 1.43, 1.25, 1.11]

    if os.path.isfile('parameter_guidelines/evaluation/german_credit/abs_horizontal_attack_utility_loss_gb_fpattr{}.pickle'.format(n_attr)):
        with open(
                'parameter_guidelines/evaluation/german_credit/abs_horizontal_attack_utility_loss_gb_fpattr{}.pickle'.format(
                        n_attr), 'rb') as infile:
            abs_horizontal_utility_loss_fpattr = pickle.load(infile)
    else:
        abs_horizontal_utility_loss_fpattr = dict()

    if os.path.isfile('parameter_guidelines/evaluation/german_credit/rel_horizontal_attack_utility_loss_gb_fpattr{}.pickle'.format(n_attr)):
        with open(
                'parameter_guidelines/evaluation/german_credit/rel_horizontal_attack_utility_loss_gb_fpattr{}.pickle'.format(
                        n_attr), 'rb') as infile:
            rel_horizontal_utility_loss_fpattr = pickle.load(infile)
    else:
        rel_horizontal_utility_loss_fpattr = dict()
    #abs_horizontal_utility_loss_fpattr = dict()
    #rel_horizontal_utility_loss_fpattr = dict()
    for gamma in gammae:
        abs_attack_utility, rel_attack_utility = horizontal_attack(gamma=gamma, n_attributes=n_attr)  # fold-wise; fingerprinted-data-wise
        abs_horizontal_utility_loss_fpattr[gamma] = abs_attack_utility
        rel_horizontal_utility_loss_fpattr[gamma] = rel_attack_utility

    with open(
            'parameter_guidelines/evaluation/german_credit/abs_horizontal_attack_utility_loss_gb_fpattr{}.pickle'.format(
                    n_attr), 'wb') as outfile:
        pickle.dump(abs_horizontal_utility_loss_fpattr, outfile)
    with open(
            'parameter_guidelines/evaluation/german_credit/rel_horizontal_attack_utility_loss_gb_fpattr{}.pickle'.format(
                    n_attr), 'wb') as outfile:
        pickle.dump(rel_horizontal_utility_loss_fpattr, outfile)

    pprint(abs_horizontal_utility_loss_fpattr)
    pprint(rel_horizontal_utility_loss_fpattr)


if __name__ == '__main__':
    #gammae = [1, 1.11, 1.25, 1.43, 1.67, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18]
    #gammae = [2.5, 1.67, 1.43, 1.25, 1.11]
    #n_attributess = [20, 16, 12, 8, 4]
    #with open('parameter_guidelines/evaluation/german_credit/abs_vertical_attack_utility_loss_gb_fpattr4.pickle', 'rb') as infile:
    #    abs_vertical_utility_loss_fpattr = pickle.load(infile)
    #with open('parameter_guidelines/evaluation/german_credit/rel_vertical_attack_utility_loss_gb_fpattr4.pickle', 'rb') as infile:
    #    rel_vertical_utility_loss_fpattr = pickle.load(infile)
    #abs_vertical_utility_loss_fpattr = dict()
    #rel_vertical_utility_loss_fpattr = dict()

    #for gamma in gammae:
    #    abs_attack_utility, rel_attack_utility = vertical_attack(gamma=gamma, n_attributes=20)  # fold-wise; fingerprinted-data-wise
    #    abs_vertical_utility_loss_fpattr[gamma] = abs_attack_utility
    #    rel_vertical_utility_loss_fpattr[gamma] = rel_attack_utility

    #with open('parameter_guidelines/evaluation/german_credit/abs_vertical_attack_utility_loss_gb_fpattr20.pickle', 'wb') as outfile:
    #    pickle.dump(abs_vertical_utility_loss_fpattr, outfile)
    #with open('parameter_guidelines/evaluation/german_credit/rel_vertical_attack_utility_loss_gb_fpattr20.pickle', 'wb') as outfile:
    #    pickle.dump(rel_vertical_utility_loss_fpattr, outfile)

    #pprint(abs_vertical_utility_loss_fpattr)
    #pprint(rel_vertical_utility_loss_fpattr)
    run_flipping(n_attr=20)
    #run_horizontal(n_attr=20)
    #print(vertical_attack(gamma=2.5, n_attributes=20))
