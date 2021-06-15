import matplotlib.pyplot as plt
import pandas as pd
import os
from pprint import pprint
from utilities import *
from scheme import AKScheme, Universal
import numpy as np
from attacks import *
from datasets import *
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
def inverse_robustness(attack, scheme,
                       primary_key_attribute=None, exclude=None, n_experiments=100, confidence_rate=0.99,
                       attack_granularity=0.10):
    attack_strength = 0
    # attack_strength = attack.get_strongest(attack_granularity)  # this should return 0+attack_granularity in case of horizontal subset attack
    # attack_strength = attack.get_weaker(attack_strength, attack_granularity)
    while True:
        if isinstance(attack, VerticalSubsetAttack):
            attack_strength += 1
        else:
            attack_strength += attack_granularity  # lower the strength of the attack
            if round(attack_strength, 2) >= 1.0:
                break
        robust = True
        success = n_experiments
        for exp_idx in range(n_experiments):
            # insert the data
            user = 1
            sk = exp_idx
            #fingerprinted_data = scheme.insertion(data, user, secret_key=sk, exclude=exclude,
            #                                      primary_key_attribute=primary_key_attribute)
            fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/adult/universal_g{}_x{}_l{}_u{}_sk{}.csv'.format(scheme.get_gamma(), 1,
                                                                                               scheme.get_fplen(),
                                                                                               user, sk))
            if isinstance(attack, VerticalSubsetAttack):
                attacked_data = attack.run_random(fingerprinted_data, attack_strength, seed=sk)
                if attacked_data is None:
                    break  # the strongest attack has been reached
            else:
                attacked_data = attack.run(fingerprinted_data, attack_strength, random_state=sk)

            # try detection
            suspect = scheme.detection(attacked_data, sk, exclude=exclude, primary_key_attribute=primary_key_attribute)

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


def get_basic_utility(original_data, fingerprinted_data):
    '''
    Gets the simple statistics for the fingerprinted dataset
    :param original_data: pandas DataFrame object
    :param fingerprinted_data: pandas DataFrame object
    :return: dictionaries of %change, mean and variance per attribute
    '''
    modification_percentage = {}
    for index in range(len(original_data.columns)):
        original = original_data[original_data.columns[index]]
        fingerprinted = fingerprinted_data[original_data.columns[index]]
        num_of_changes = len(original.compare(fingerprinted))
        modification_percentage[original_data.columns[index]] = (num_of_changes / len(original_data)) * 100

    mean_original = [np.mean(original_data[attribute]) for attribute in original_data]
    mean_fingerprint = [np.mean(fingerprinted_data[attribute]) for attribute in fingerprinted_data]
    delta_mean = {attribute: fp - org for attribute, fp, org in zip(original_data, mean_fingerprint, mean_original)}

    var_original = [np.var(original_data[attribute]) for attribute in original_data]
    var_fingerprint = [np.var(fingerprinted_data[attribute]) for attribute in fingerprinted_data]
    delta_var = {attribute: fp - org for attribute, fp, org in zip(original_data, var_fingerprint, var_original)}

    return modification_percentage, delta_mean, delta_var


# runs deterministic experiments on data utility via KNN
def attack_utility_knn(data, target, attack, attack_granularity=0.1, n_folds=10):
    X = data.drop(target, axis=1)
    y = data[target]

    attack_strength = 0
    utility = dict()
    while True:
        attack_strength += attack_granularity  # lower the strength of the attack
        if round(attack_strength, 2) >= 1.0:
            break
        # score = cross_val_score(model, X, y, cv=5)
        accuracy = []
        for fold in range(n_folds):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=fold, shuffle=True)
            train = pd.concat([X_train, y_train], axis=1)
            attacked_train = attack.run(train, attack_strength, random_state=fold)
            attacked_X = attacked_train.drop(target, axis=1)
            attacked_y = attacked_train[target]

            model = KNeighborsClassifier()
            model.fit(attacked_X, attacked_y)
            acc = accuracy_score(y_test, model.predict(X_test))
            accuracy.append(acc)
        utility[round(1-attack_strength, 2)] = accuracy
    return utility
    # returns estimated utility drop for each attack strength


def fingerprint_utility_knn(data, target, gamma, n_folds=10, n_experiments=10, data_string=None):
    # n_folds should be consistent with experiments done on attacked data
    if isinstance(data, Dataset):
        data_string = data.to_string()
        data = data.preprocessed()
    X = data.drop(target, axis=1)
    y = data[target]
    model = KNeighborsClassifier()

    fingerprinted_data_dir = 'parameter_guidelines/fingerprinted_data/{}/'.format(data_string)

    accuracy = []
    for exp in range(n_experiments):
        fp_file_string = 'universal_g{}_x1_l32_u1_sk{}.csv'.format(gamma, exp)
        fingerprinted_data = pd.read_csv(fingerprinted_data_dir+fp_file_string)
        fingerprinted_data = Adult().preprocessed(fingerprinted_data)
        X_fp = fingerprinted_data.drop(target, axis=1)
        y_fp = fingerprinted_data[target]

        acc = fp_cross_val_score(model, X, y, X_fp, y_fp, cv=n_folds, scoring='accuracy')['test_score']
        accuracy.append(acc)

    # [[acc_fold1,acc_fold2,...],[],...n_experiments]
    return accuracy


def original_utility_knn(data, target, n_folds=10):
    # n_folds should be consistent with experiments done on attacked data
    X = data.drop(target, axis=1)
    y = data[target]

    accuracy = []
    for fold in range(n_folds):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=fold, shuffle=True)

        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        accuracy.append(acc)
    return accuracy


def original_utility_dt(data, target, n_folds=10):
    # n_folds should be consistent with experiments done on attacked data
    X = data.drop(target, axis=1)
    y = data[target]

    accuracy = []
    for fold in range(n_folds):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=fold, shuffle=True)

        model = DecisionTreeClassifier(random_state=0)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        accuracy.append(acc)
    return accuracy


def original_utility_gb(data, target, n_folds=10):
    # n_folds should be consistent with experiments done on attacked data
    X = data.drop(target, axis=1)
    y = data[target]

    accuracy = []
    for fold in range(n_folds):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=fold, shuffle=True)

        model = GradientBoostingClassifier(random_state=0)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        accuracy.append(acc)
    return accuracy


def fingerprint_utility_dt(data, target, gamma, n_folds=10, n_experiments=10, data_string=None):
    # n_folds should be consistent with experiments done on attacked data
    if isinstance(data, Dataset):
        data_string = data.to_string()
        data = data.preprocessed()
    X = data.drop(target, axis=1)
    y = data[target]

    fingerprinted_data_dir = 'parameter_guidelines/fingerprinted_data/{}/'.format(data_string)

    accuracy = []
    for exp in range(n_experiments):
        fp_file_string = 'universal_g{}_x1_l32_u1_sk{}.csv'.format(gamma, exp)
        fingerprinted_data = pd.read_csv(fingerprinted_data_dir+fp_file_string)
        fingerprinted_data = Adult().preprocessed(fingerprinted_data)
        X_fp = fingerprinted_data.drop(target, axis=1)
        y_fp = fingerprinted_data[target]

        model = DecisionTreeClassifier(random_state=0)
        acc = fp_cross_val_score(model, X, y, X_fp, y_fp, cv=n_folds, scoring='accuracy')['test_score']
        accuracy.append(acc)

    # [[acc_fold1,acc_fold2,...],[],...n_experiments]
    return accuracy


def fingerprint_utility_gb(data, target, gamma, n_folds=10, n_experiments=10, data_string=None):
    # n_folds should be consistent with experiments done on attacked data
    if isinstance(data, Dataset):
        data_string = data.to_string()
        data = data.preprocessed()
    X = data.drop(target, axis=1)
    y = data[target]

    fingerprinted_data_dir = 'parameter_guidelines/fingerprinted_data/{}/'.format(data_string)

    accuracy = []
    for exp in range(n_experiments):
        fp_file_string = 'universal_g{}_x1_l32_u1_sk{}.csv'.format(gamma, exp)
        fingerprinted_data = pd.read_csv(fingerprinted_data_dir + fp_file_string)
        fingerprinted_data = Adult().preprocessed(fingerprinted_data)
        X_fp = fingerprinted_data.drop(target, axis=1)
        y_fp = fingerprinted_data[target]

        model = GradientBoostingClassifier(random_state=0)
        acc = fp_cross_val_score(model, X, y, X_fp, y_fp, cv=n_folds, scoring='accuracy')['test_score']
        accuracy.append(acc)

    # [[acc_fold1,acc_fold2,...],[],...n_experiments]
    return accuracy


def attack_utility_bounds(original_utility, attack_utility):
    # returns the attack strengths that yield at least 1%, 2%, ... utility loss and
    # the largest accuracy drop of attacked data
    # the returned values are absolute
    attack_bounds = []
    max_utility_drop = np.mean(original_utility) - \
                       min(np.mean(acc) for acc in attack_utility.values())
    drop = 0.01
    while drop < max_utility_drop:
        # attack strength that yields at least 1%(or p%) of accuracy loss

        attack_strength = max([strength for strength in attack_utility
                          if np.mean(original_utility) - np.mean(attack_utility[strength])
                               <= drop])
        attack_bounds.append(attack_strength)
        drop += 0.01
    attack_bounds.append(max_utility_drop)
    return attack_bounds


#def _split_features_target(original_data, fingerprinted_data):
#    X = data.drop([target, 'sample-code-number'], axis=1)
#    y = data[target]


#def get_ML_utility():
#    # todo: also baseline models (original) should be done only once
#    X, y, X_fp, y_fp = _split_features_target(original_data, fingerprinted_data)
#    _utility_KNN()


def master_evaluation(dataset,
                      target_attribute=None, primary_key_attribute=None):
    '''
    This method outputs the full robustness and utility evaluation to user 'at glance', given the data set.
    This includes: (1) utility approximation trends and (2) expected robustness trends
    The outputs should help the user with parameter choices for their data set.

    (1) Utility evaluation shows (i) the average change in mean and variance for each attribute and (ii) average
    performance of the fingerprinted data sets using a variety of classifiers, e.g. Decision Tree,
    Logistic Regression, Gradient Boosting...
    :param dataset: path to the dataset, pandas DataFrame or class Dataset
    :param target_attribute: name of the target attribute for the dataset. Ignored if dataset is of type Dataset
    :param primary_key_attribute: name of the primary key attribute of the dataset. Ignored if dataset is of type Dataset
    :return: metadata of the experimental run
    '''
    meta = ''
    if isinstance(dataset, str):  # assumed the path is given
        data = Dataset(path=dataset, target_attribute=target_attribute, primary_key_attribute=primary_key_attribute)
    elif isinstance(dataset, pd.DataFrame):  # assumed the pd.DataFrame is given
        data = Dataset(dataframe=dataset, target_attribute=target_attribute, primary_key_attribute=primary_key_attribute)
    elif isinstance(dataset, Dataset):
        data = dataset
    else:
        print('Wrong type of input data.')
        exit()

    # EXPERIMENT RUN
    # 1) fingerprint the data (i.e. distinct secret key & distinct gamma)
    # 2) record the changes in mean and variance for each attribute
    # 3) perform the classification analysis
    # 4) robustness per se (extraction rate)
    # 5) robustness against the attacks (experimental) -> here it would make sense to compare the theoretical results

    _start_exp_run = time.time()

    # todo: for now only integer data fingerprinting is supported via AK scheme. Next up: categorical & decimal
    gamma = 2
    secret_key = 123
    scheme = AKScheme(gamma=gamma, fingerprint_bit_length=16)

    fingerprinted_data = scheme.insertion(dataset=data, secret_key=secret_key, recipient_id=0)

    changed_vals, mean, var = get_basic_utility(data.get_dataframe(), fingerprinted_data.get_dataframe())

#    get_ML_utility()

    return meta


def test_interactive_plot():
    fig, ax = plt.subplots(figsize=(14, 8))

    y = np.random.randint(0, 100, size=50)
    x = np.random.choice(np.arange(len(y)), size=10)

    line, = ax.plot(y, '-', label='line')
    dot, = ax.plot(x, y[x], 'o', label='dot')

    legend = plt.legend(loc='upper right')
    line_legend, dot_legend = legend.get_lines()
    line_legend.set_picker(True)
    line_legend.set_pickradius(10)
    dot_legend.set_picker(True)
    dot_legend.set_pickradius(10)

    graphs = {}
    graphs[line_legend] = line
    graphs[dot_legend] = dot

    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()

        graphs[legend].set_visible(not isVisible)
        legend.set_visible(not isVisible)

        fig.canvas.draw()

    plt.connect('pick_event', on_pick)
    plt.show()


if __name__ == '__main__':
    test_interactive_plot()
