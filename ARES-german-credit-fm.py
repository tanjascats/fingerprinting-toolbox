import random
import pandas as pd
import math
import time
from hashlib import blake2b
from bitstring import BitArray
import sys
from attacks import *
from scheme import *
from datasets import *
import pickle


#GAMMA = 2
#XI = 2
SECRET_KEY = 3476  #670  #3476  #670

FINGERPRINT_BIT_LENGTH = 64
NUMBER_OF_BUYERS = 10

def false_miss(attack, attack_strength, scheme, data, exclude=None, include=None, n_experiments=100, confidence_rate=0.99,
               attack_granularity=0.10):
    #attack_strength = 1  # defining the strongest attack
    #attack_vertical_max = -1
    # attack_strength = attack.get_strongest(attack_granularity)  # this should return 0+attack_granularity in case of horizontal subset attack
    # attack_strength = attack.get_weaker(attack_strength, attack_granularity)
    #while True:
        #if isinstance(attack, VerticalSubsetAttack):
        #    attack_strength -= 1  # lower the strength of the attack
        #    if attack_strength == 0 and attack_vertical_max != -1:
        #        break
        #else:
        #    # how much data will stay in the release, not how much it will be deleted
        #    attack_strength -= attack_granularity  # lower the strength of the attack
        #    if round(attack_strength, 2) <= 0:  # break if the weakest attack is reached
        #        break
        #robust = True  # for now it's robust
    success = n_experiments
    for exp_idx in range(n_experiments):
        # insert the data
        user = 1
        sk = exp_idx
        #fingerprinted_data = scheme.insertion(data, user, secret_key=sk, exclude=exclude,
        #                                      primary_key_attribute=primary_key_attribute)
        if include is None:
            fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/' + data.to_string() +
                                             '/attr_subset_20' +
                                             '/universal_g{}_x{}_l{}_u{}_sk{}.csv'.format(scheme.get_gamma(), 1,
                                                                                               scheme.get_fplen(),
                                                                                               user, sk))
        else:
            fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/' + data.to_string() +
                                             '/attr_subset_' + str(len(include)) +
                                             '/universal_g{}_x{}_l{}_u{}_sk{}.csv'.format(scheme.get_gamma(), 1,
                                                                                          scheme.get_fplen(),
                                                                                          user, sk))
        if isinstance(attack, VerticalSubsetAttack):
            #if attack_vertical_max == -1:  # remember the strongest attack and initiate the attack strength
            #    attack_vertical_max = len(fingerprinted_data.columns.drop([data.get_target_attribute(),
            #                                                               data.get_primary_key_attribute()]))
            #   attack_strength = attack_vertical_max - 1
            attacked_data = attack.run_random(dataset=fingerprinted_data, number_of_columns_to_del=attack_strength, seed=sk)
        else:
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
            #original_attributes = pd.Series(
            #    data=['checking_account', 'duration', 'credit_hist', 'purpose',
            #          'credit_amount', 'savings', 'employment_since', 'installment_rate',
            #          'sex_status', 'debtors', 'residence_since', 'property', 'age',
            #          'installment_other', 'housing', 'existing_credits', 'job',
            #          'liable_people', 'tel', 'foreign'])
            original_attributes= pd.Index(['checking_account', 'duration', 'credit_hist', 'purpose',
                   'credit_amount', 'savings', 'employment_since', 'installment_rate',
                   'sex_status', 'debtors', 'residence_since', 'property', 'age',
                   'installment_other', 'housing', 'existing_credits', 'job',
                   'liable_people', 'tel', 'foreign'],
                  dtype='object')
        suspect = scheme.detection(attacked_data, secret_key=sk, primary_key_attribute='Id',target_attribute='target',
                                   original_attributes=original_attributes
        )

        if suspect != user:
            success -= 1

            #if success / n_experiments < confidence_rate:
            #    robust = False
            #    print('-------------------------------------------------------------------')
            #    print('-------------------------------------------------------------------')
            #    print(
            #        'Attack with strength ' + str(attack_strength) + " is too strong. Halting after " + str(exp_idx) +
            #        " iterations.")
            #    print('-------------------------------------------------------------------')
            #    print('-------------------------------------------------------------------')
            #    break  # attack too strong, continue with a lighter one
        #if robust:
        #    if isinstance(attack, VerticalSubsetAttack):
        #        attack_strength = round(attack_strength / attack_vertical_max, 2)
        #    return round(attack_strength, 2)
    #if isinstance(attack, VerticalSubsetAttack):
    #    attack_strength = round(attack_strength / attack_vertical_max, 2)
    # todo: mirror the performance for >0.5 flipping attacks
    return round(1.0 - success/n_experiments, 2)


## from eval/robustness/bit-flip_attack.py
def main():
    gamma=1
    xi=1
    fplen=8
    numbuyers=100
    attack = FlippingAttack()
    data = GermanCredit()
    target='target'

    # from how much remaining data can the fingerprint still be extracted?
    fm = dict()
    for gamma in [1, 3, 6, 12]:
        scheme = Universal(gamma=gamma,
                           xi=xi,
                           fingerprint_bit_length=fplen,
                           number_of_recipients=numbuyers)
        fm[gamma] = dict()
        for attack_strength in [0]: #, 6, 12, 18, 19]:
            fm[gamma][attack_strength] = false_miss(attack=attack, attack_strength=attack_strength, scheme=scheme, data=data,
                                                    n_experiments=1)
    print(fm)
    with open('fm_gc_vertical_2.pkl', 'wb') as outfile:
        pickle.dump(fm, outfile)

## valid experiments with Vertical attack
def test_vertical():
    n_exp = 100
    gamma = 3
    scheme = Universal(gamma=gamma,
                       xi=1,
                       fingerprint_bit_length=8,
                       number_of_recipients=100)
    data = GermanCredit()
    attack = VerticalSubsetAttack()
    strength = int(0.95*20)

    success = 0
    for exp in range(n_exp):
        print('#exp ', exp)
        fp_data = scheme.insertion(dataset=data, secret_key=exp, recipient_id=1)
        attacked = attack.run_random(dataset=fp_data.dataframe, number_of_columns_to_del=strength, seed=exp,
                                     keep_columns=['target'])
        recipient = scheme.detection(attacked, secret_key=exp, target_attribute='target', primary_key_attribute='Id')
        if recipient == 1:
            success += 1
    print('gamma =', gamma)
    print('strength = ', strength)
    print('fm =', round(1.0 - success/n_exp, 2))


def test_flipping():
    gamma = 3
    scheme = Universal(gamma=gamma,
                       xi=1,
                       fingerprint_bit_length=8,
                       number_of_recipients=100)
    data = GermanCredit()
    attack = FlippingAttack()
    strength = 0.95

    n_exp = 100

    success = 0
    for exp in range(n_exp):
        fp_data = scheme.insertion(dataset=data, secret_key=exp, recipient_id=1)
        attacked = attack.run(dataset=fp_data.dataframe, strength=strength, keep_columns=['target'], random_state=exp)
        recipient = scheme.detection(attacked, secret_key=exp, target_attribute='target', primary_key_attribute='Id')
        if recipient == 1:
            success += 1
    print('gamma =', gamma)
    print('strength = ', strength)
    print('fm =', round(1.0 - success / n_exp, 2))


def test_horizontal():
    gamma = 1
    scheme = Universal(gamma=gamma,
                       xi=1,
                       fingerprint_bit_length=8,
                       number_of_recipients=100)
    data = GermanCredit()
    attack = HorizontalSubsetAttack()
    strength = 0.95

    n_exp = 100

    success = 0
    for exp in range(n_exp):
        fp_data = scheme.insertion(dataset=data, secret_key=exp, recipient_id=1)
        attacked = attack.run(dataset=fp_data.dataframe, strength=strength, random_state=exp)
        recipient = scheme.detection(attacked, secret_key=exp, target_attribute='target',
                                     primary_key_attribute='Id')
        if recipient == 1:
            success += 1
    print('gamma =', gamma)
    print('strength = ', strength)
    print('fm =', round(1.0 - success / n_exp, 2))


if __name__ == '__main__':
    #main()
    #test_vertical()
    test_horizontal()
