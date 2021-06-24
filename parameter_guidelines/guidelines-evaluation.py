import os
os.chdir("C:\\Users\\tsarcevic\\PycharmProjects\\fingerprinting-toolbox")

import pickle
from parameter_guidelines.guidelines import *


# -------------------------------------------------#
# ROBUSTNESS EVALUATION #
# ------------------------------------------------ #
def robustness_evaluation(attack, data, attack_string, gammae, attack_granularity=0.05, n_experiments=100,
                          confidence_rate=0.9,
                          target=None):
    try:
        target = data.get_target_attribute()
    except AttributeError:
        pass

    xi = 1
    fplen = 32
    numbuyers = 100

    if attack_string == 'vertical':
        file_string = 'robustness_{}_universal_c{}_e{}.pickle'.format(attack_string,
                                                                      format(confidence_rate, ".2f")[-2:],
                                                                      n_experiments)
    else:
        file_string = 'robustness_{}_universal_c{}_ag{}_e{}.pickle'.format(attack_string,
                                                                           format(confidence_rate, ".2f")[-2:],
                                                                           format(attack_granularity, ".2f")[-2:],
                                                                           n_experiments)
    # check if results exist
    # ---------------------- #
    if os.path.isfile('parameter_guidelines/evaluation/adult/' + file_string):
        with open('parameter_guidelines/evaluation/adult/' + file_string, 'rb') as infile:
            resutls = pickle.load(infile)
    else:
        resutls = {}
    gammae_new = []
    for gamma in gammae:
        if gamma not in resutls.keys():
            gammae_new.append(gamma)
            print('Updating results with gamma={}'.format(gamma))
    # ---------------------- #

    for gamma in gammae_new:
        scheme = Universal(gamma=gamma,
                           xi=xi,
                           fingerprint_bit_length=fplen,
                           number_of_recipients=numbuyers)
        # from how much remaining data can the fingerprint still be extracted?
        remaining = robustness(attack, scheme, exclude=[target],
                               attack_granularity=attack_granularity,
                               n_experiments=n_experiments,
                               confidence_rate=confidence_rate)
        resutls[gamma] = remaining
    resutls = dict(sorted(resutls.items()))
    with open('parameter_guidelines/evaluation/adult/' + file_string, 'wb') as outfile:
        pickle.dump(resutls, outfile)


if __name__ == '__main__':
    data = pd.read_csv('datasets/adult.csv', na_values='?')
    # # to focus on real stuff, let's ignore missing values
    data = data.dropna()
    xi = 1
    fplen = 32
    numbuyers = 100
    gammae = [1, 2, 3, 4, 5, 6, 10, 12, 15, 18, 20, 25, 30, 50, 100, 200]  # 35 40 50 60 70 80
    # data = Adult().preprocessed()

    attack = FlippingAttack()
    robustness_evaluation(attack, data, 'flipping', gammae=[1,2,3,6,12,50], confidence_rate=0.95, target='income',
                          n_experiments=100,
                          attack_granularity=0.05)

