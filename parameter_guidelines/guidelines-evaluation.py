import os
os.chdir("C:\\Users\\tsarcevic\\PycharmProjects\\fingerprinting-toolbox")

import pickle
from parameter_guidelines.guidelines import *
from scheme import *


# -------------------------------------------------#
# ROBUSTNESS EVALUATION #
# ------------------------------------------------ #
def robustness_evaluation(attack, data, attack_string, gammae, attack_granularity=0.05, n_experiments=100,
                          confidence_rate=0.9,
                          target=None,
                          attribute_subset=None):
    # attribute subset is expected to be a list of attributes included in the fingerprinting
    try:
        target = data.get_target_attribute()
    except AttributeError:
        pass

    xi = 1
    fplen = 32
    numbuyers = 100

    if attack_string == 'vertical':
        if attribute_subset is None:
            file_string = 'robustness_{}_universal_c{}_e{}.pickle'.format(attack_string,
                                                                          format(confidence_rate, ".2f")[-2:],
                                                                          n_experiments)
        else:
            file_string = 'robustness_{}_universal_c{}_fpattr{}_e{}.pickle'.format(attack_string,
                                                                          format(confidence_rate, ".2f")[-2:],
                                                                          len(attribute_subset),
                                                                                   n_experiments)
    else:
        if attribute_subset is None:
            file_string = 'robustness_{}_universal_c{}_ag{}_e{}.pickle'.format(attack_string,
                                                                               format(confidence_rate, ".2f")[-2:],
                                                                               format(attack_granularity, ".2f")[-2:],
                                                                               n_experiments)
        else:
            file_string = 'robustness_{}_universal_c{}_ag{}_fpattr{}_e{}.pickle'.format(attack_string,
                                                                               format(confidence_rate, ".2f")[-2:],
                                                                               format(attack_granularity, ".2f")[-2:],
                                                                                        len(attribute_subset),
                                                                                        n_experiments)
    # check if results exist
    # ---------------------- #
    if os.path.isfile('parameter_guidelines/evaluation/' + data.to_string() + "/" + file_string):
        with open('parameter_guidelines/evaluation/' + data.to_string() + "/" + file_string, 'rb') as infile:
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
        remaining = robustness(attack, scheme, data, exclude=[target],
                               include=attribute_subset,
                               attack_granularity=attack_granularity,
                               n_experiments=n_experiments,
                               confidence_rate=confidence_rate)
        resutls[gamma] = remaining
    resutls = dict(sorted(resutls.items()))
    # todo: remove comment
    print(resutls)
    with open('parameter_guidelines/evaluation/' + data.to_string() + "/" + file_string, 'wb') as outfile:
        pickle.dump(resutls, outfile)


if __name__ == '__main__':
    data = Nursery()
    xi = 1
    fplen = 32
    numbuyers = 100
    #gammae = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18]  # 35 40 50 60 70 80
    gammae = [2.5, 1.67, 1.43, 1.25, 1.11] #[2.5]
    gammae = [1]
    # data = Adult().preprocessed()

    attack = VerticalSubsetAttack()
    robustness_evaluation(attack, data, 'vertical', gammae=gammae, confidence_rate=0.95,
                          n_experiments=100,
                          attack_granularity=0.05
#                          ,
#                          attribute_subset=['checking_account', 'duration', 'credit_hist', 'purpose'
#                                            ,'credit_amount', 'savings','employment_since', 'installment_rate'
#                                            ,'sex_status','debtors','residence_since','property'
#                                            ,'age','installment_other','housing','existing_credits'
#                                            ,'job', 'liable_people', 'tel', 'foreign'
#    ]
    )
