from parameter_guidelines.guidelines import *
from scheme import *


def test1():
    # original data
    data = GermanCredit()
    X = data.preprocessed().drop('target', axis=1)
    y = data.preprocessed()['target']

    # embedding the fingerpting - test case
    fplen = 8
    numbuyers = 100
    # column_subset = 16
    gamma = 1
    xi = 1
    uid = 1
    SK = 0

    scheme = Universal(gamma=gamma, xi=xi, fingerprint_bit_length=fplen, number_of_recipients=100)
    exclude = ['sex_status', 'installment_other', 'housing', 'foreign']
    fingerprinted_data = scheme.insertion(data, recipient_id=uid, secret_key=SK, exclude=exclude,
                                          primary_key_attribute='Id',
                                          target_attribute='target')

    # test detection
    scheme = Universal(gamma=gamma, xi=xi, fingerprint_bit_length=fplen, number_of_recipients=100)
    suspect = scheme.detection(fingerprinted_data, secret_key=SK, target_attribute='target', primary_key_attribute='Id',
                               exclude=exclude, original_attributes=pd.Series(data=X.columns.to_list()))
    print(suspect)


def drop_least_important(n, features):
    remaining = features
    for i in range(n):
        min_val = min(remaining.values())
        remaining = {k: v for k, v in remaining.items() if v != min_val}
    return remaining


def test2():
    # find strength that removes the fingerprint
    attack_strength = 10
    # attacker's features
    feature_importances_attack = {'checking_account': 0.21514958469895673,
                                  'duration': 0.13044392484665898,
                                  'credit_hist': 0.08577361202438204,
                                  'purpose': 0.037542360031586446,
                                  'credit_amount': 0.17261276703874145,
                                  'savings': 0.04680897088703411,
                                  'employment_since': 0.03161875938692153,
                                  'installment_rate': 0.022951027357519246,
                                  'sex_status': 0.006977414324585844,
                                  'debtors': 0.01549920743767321,
                                  'residence_since': 0.006314167794645683,
                                  'property': 0.04161068704700113,
                                  'age': 0.09356728372727559,
                                  'installment_other': 0.04487824182105095,
                                  'housing': 0.010520190596630376,
                                  'existing_credits': 0.013612419726994892,
                                  'job': 0.008691845710110693,
                                  'liable_people': 0.010437271782373785,
                                  'tel': 0.003110805448656495,
                                  'foreign': 0.0018794583112007593}
    # original data
    data = GermanCredit()
    X = data.preprocessed().drop('target', axis=1)
    y = data.preprocessed()['target']
    X.columns
    # embedding the fingerpting - test case
    fplen = 8
    numbuyers = 100
    column_subset = 20
    gamma = 18
    xi = 1
    uid = 1
    SK = 0

    scheme = Universal(gamma=gamma, xi=xi, fingerprint_bit_length=fplen, number_of_recipients=100)
    exclude = ['sex_status', 'installment_other', 'housing', 'foreign']
    fingerprinted_data = scheme.insertion(data, recipient_id=uid, secret_key=SK, exclude=exclude,
                                          primary_key_attribute='Id',
                                          target_attribute='target')
    # dict(sorted(feature_importances_attack.items(), key=lambda item: -item[1]))
    selected_f = drop_least_important(attack_strength, feature_importances_attack)
    removed = list(feature_importances_attack.keys() - selected_f.keys())
    attacked_data = fingerprinted_data.dataframe.drop(removed, axis=1)
    suspect = scheme.detection(attacked_data, secret_key=SK, target_attribute='target', primary_key_attribute='Id',
                               exclude=exclude, original_attributes=pd.Series(data=X.columns.to_list()))


if __name__ == '__main__':

    test2()
