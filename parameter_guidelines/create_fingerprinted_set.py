from utilities import *
from scheme import Universal
from datasets import *
import os.path
import glob
import random


def german_credit():
    data = GermanCredit()
    gammae = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15,
              18]  # [2.5, 1.67, 1.43, 1.25, 1.11] #[2.5]#[1,2,3,4,5,6,7,8,9,10,12,15,18]#,20,25,30,35,40,50]
    fplen = 8
    numbuyers = 100
    # exclude = [data.get_target_attribute(), data.get_primary_key_attribute()]
    for g in gammae:
        for sk in range(100):
            column_subset = 4
            file_string = 'parameter_guidelines/fingerprinted_data/' + data.to_string() + \
                          '/attr_subset_' + str(column_subset) + \
                          '/universal_g{}_x{}_l{}_u{}_sk{}.csv'.format(g, 1, fplen, 1, sk)  # u -> user
            if not os.path.isfile(file_string):
                scheme = Universal(gamma=g, xi=1, fingerprint_bit_length=fplen, number_of_recipients=numbuyers)
                # define exclude param for fingerprinting a subset of columns
                exclude = ['foreign', 'liable_people', 'tel', 'job'
                    , 'age', 'installment_other', 'housing', 'existing_credits'
                    , 'sex_status', 'debtors', 'residence_since', 'property'
                    , 'credit_amount', 'savings', 'employment_since', 'installment_rate'
                           ]
                fingerprinted_data = scheme.insertion(data, recipient_id=1, secret_key=sk, exclude=exclude,
                                                      write_to=file_string)
                fingerprinted_data.get_dataframe().to_csv(file_string, index=False)
            else:
                print('File already exists; skipping gamma={}, no.{}'.format(g, sk))


def adult():
    # data = pd.read_csv('datasets/adult.csv', na_values='?')
    # data = data.dropna(axis=0).reset_index().drop('index', axis=1)
    data = Adult()
    gammae = [1.11, 1.25, 1.43, 1.67, 2.5]  #[2.5]#[1,2,3,4,5,6,7,8,9,10,12,15,18]#,20,25,30,35,40,50]
    fplen = 32
    numbuyers = 100
    # exclude = [data.get_target_attribute(), data.get_primary_key_attribute()]
    for g in gammae:
        for sk in range(30):
            column_subset = 12
            file_string = 'parameter_guidelines/fingerprinted_data/' + data.to_string() + \
                          '/attr_subset_' + str(column_subset) + \
                          '/universal_g{}_x{}_l{}_u{}_sk{}.csv'.format(g, 1, fplen, 1, sk)  # u -> user
            if not os.path.isfile(file_string):
                scheme = Universal(gamma=g, xi=1, fingerprint_bit_length=fplen, number_of_recipients=numbuyers)
                # define exclude param for fingerprinting a subset of columns
                exclude = ['education', 'fnlwgt']
                fingerprinted_data = scheme.insertion(data, recipient_id=1, secret_key=sk, exclude=exclude,
                                                      write_to=file_string)
                fingerprinted_data.get_dataframe().to_csv(file_string, index=False)
            else:
                print('File already exists; skipping gamma={}, no.{}'.format(g, sk))


def bank_personal_loan():
    data = BankPersonalLoan()
    gammae = [1.11, 1.25, 1.43, 1.67, 2.5]  # [2.5, 1.67, 1.43, 1.25, 1.11] #[2.5]#[1,2,3,4,5,6,7,8,9,10,12,15,18]#,20,25,30,35,40,50]
    fplen = 16
    numbuyers = 100
    # exclude = [data.get_target_attribute(), data.get_primary_key_attribute()]
    for g in gammae:
        for sk in range(100):
            column_subset = 12
            file_string = 'parameter_guidelines/fingerprinted_data/' + data.to_string() + \
                          '/attr_subset_' + str(column_subset) + \
                          '/universal_g{}_x{}_l{}_u{}_sk{}.csv'.format(g, 1, fplen, 1, sk)  # u -> user
            if not os.path.isfile(file_string):
                scheme = Universal(gamma=g, xi=1, fingerprint_bit_length=fplen, number_of_recipients=numbuyers)
                # define exclude param for fingerprinting a subset of columns
                fingerprinted_data = scheme.insertion(data, recipient_id=1, secret_key=sk,
                                                      write_to=file_string)
                fingerprinted_data.get_dataframe().to_csv(file_string, index=False)
            else:
                print('File already exists; skipping gamma={}, no.{}'.format(g, sk))


def breast_cancer():
    data = BreastCancer()
    gammae = [1.11, 1.25, 1.43, 1.67, 2.5]
    # [2.5, 1.67, 1.43, 1.25, 1.11] #[2.5]#[1,2,3,4,5,6,7,8,9,10,12,15,18]#,20,25,30,35,40,50]
    gammae = [1, 2, 3, 4, 5, 10, 18]
    fplen = 8
    numbuyers = 100
    # exclude = [data.get_target_attribute(), data.get_primary_key_attribute()]
    for g in gammae:
        for sk in range(50):
            column_subset = 9
            file_string = 'parameter_guidelines/fingerprinted_data/' + data.to_string() + \
                          '/attr_subset_' + str(column_subset) + \
                          '/universal_g{}_x{}_l{}_u{}_sk{}.csv'.format(g, 1, fplen, 1, sk)  # u -> user
            if not os.path.isfile(file_string):
                scheme = Universal(gamma=g, xi=1, fingerprint_bit_length=fplen, number_of_recipients=numbuyers)
                # define exclude param for fingerprinting a subset of columns
                fingerprinted_data = scheme.insertion(data, recipient_id=1, secret_key=sk,
                                                      write_to=file_string)
                fingerprinted_data.get_dataframe().to_csv(file_string, index=False)
            else:
                print('File already exists; skipping gamma={}, no.{}'.format(g, sk))


def diabetic_data():
    data = DiabeticData()
    gammae = [1.11, 1.25, 1.43, 1.67, 2.5]
    # [2.5, 1.67, 1.43, 1.25, 1.11] #[2.5]#[1,2,3,4,5,6,7,8,9,10,12,15,18]#,20,25,30,35,40,50]
    #gammae = [1, 2, 3, 4, 5, 10, 18]
    fplen = 32
    numbuyers = 100
    # exclude = [data.get_target_attribute(), data.get_primary_key_attribute()]
    for g in gammae:
        for sk in range(20):
            column_subset = 45
            file_string = 'parameter_guidelines/fingerprinted_data/' + data.to_string() + \
                          '/attr_subset_' + str(column_subset) + \
                          '/universal_g{}_x{}_l{}_u{}_sk{}.csv'.format(g, 1, fplen, 1, sk)  # u -> user
            if not os.path.isfile(file_string):
                scheme = Universal(gamma=g, xi=1, fingerprint_bit_length=fplen, number_of_recipients=numbuyers)
                # define exclude param for fingerprinting a subset of columns
                fingerprinted_data = scheme.insertion(data, recipient_id=1, secret_key=sk,
                                                      write_to=file_string)
                fingerprinted_data.get_dataframe().to_csv(file_string, index=False)
            else:
                print('File already exists; skipping gamma={}, no.{}'.format(g, sk))


def mushrooms():
    data = Mushrooms()
    gammae = [1.11, 1.25, 1.43, 1.67, 2.5]
    # [2.5, 1.67, 1.43, 1.25, 1.11] #[2.5]#[1,2,3,4,5,6,7,8,9,10,12,15,18]#,20,25,30,35,40,50]
    #gammae = [1, 2, 3, 4, 5, 10, 18]
    fplen = 16
    numbuyers = 100
    # exclude = [data.get_target_attribute(), data.get_primary_key_attribute()]
    for g in gammae:
        for sk in range(30):
            column_subset = 22
            file_string = 'parameter_guidelines/fingerprinted_data/' + data.to_string() + \
                          '/attr_subset_' + str(column_subset) + \
                          '/universal_g{}_x{}_l{}_u{}_sk{}.csv'.format(g, 1, fplen, 1, sk)  # u -> user
            if not os.path.isfile(file_string):
                scheme = Universal(gamma=g, xi=1, fingerprint_bit_length=fplen, number_of_recipients=numbuyers)
                # define exclude param for fingerprinting a subset of columns
                fingerprinted_data = scheme.insertion(data, recipient_id=1, secret_key=sk,
                                                      write_to=file_string)
                fingerprinted_data.get_dataframe().to_csv(file_string, index=False)
            else:
                print('File already exists; skipping gamma={}, no.{}'.format(g, sk))


def nursery():
    data = Nursery()
    #gammae = [1.11, 1.25, 1.43, 1.67, 2.5]
    # [2.5, 1.67, 1.43, 1.25, 1.11] #[2.5]#[1,2,3,4,5,6,7,8,9,10,12,15,18]#,20,25,30,35,40,50]
    gammae = [1, 2, 3, 4, 5, 10, 18]
    fplen = 32
    numbuyers = 100
    # exclude = [data.get_target_attribute(), data.get_primary_key_attribute()]
    for g in gammae:
        for sk in range(30):
            column_subset = 8
            file_string = 'parameter_guidelines/fingerprinted_data/' + data.to_string() + \
                          '/attr_subset_' + str(column_subset) + \
                          '/universal_g{}_x{}_l{}_u{}_sk{}.csv'.format(g, 1, fplen, 1, sk)  # u -> user
            if not os.path.isfile(file_string):
                scheme = Universal(gamma=g, xi=1, fingerprint_bit_length=fplen, number_of_recipients=numbuyers)
                # define exclude param for fingerprinting a subset of columns
                fingerprinted_data = scheme.insertion(data, recipient_id=1, secret_key=sk,
                                                      write_to=file_string)
                fingerprinted_data.get_dataframe().to_csv(file_string, index=False)
            else:
                print('File already exists; skipping gamma={}, no.{}'.format(g, sk))


def test_detection():
    attr_subset = 4
    directoryPath = os.path.join("parameter_guidelines", "fingerprinted_data", "german_credit",
                                 "attr_subset_" + str(attr_subset))
    for file_name_full in glob.glob(directoryPath + '/*.csv'):
        file_name = file_name_full.split("\\")[-1].split('_')
        try:
            gamma = int(file_name[1][1:])
        except ValueError:
            gamma = float(file_name[1][1:])
        sk = int(file_name[-1][2:-4])
        true_recipient = int(file_name[4][1:])
        fp_len = int(file_name[3][1:])
        xi = int(file_name[2][1:])

        fp_data = pd.read_csv(file_name_full, low_memory=False)
        scheme = Universal(gamma=gamma, xi=xi, fingerprint_bit_length=fp_len, number_of_recipients=100)

        suspect = scheme.detection(fp_data, secret_key=sk, target_attribute='target', primary_key_attribute='Id',
                                   exclude=['foreign', 'liable_people', 'tel', 'job'],
                                   original_attributes=pd.Series(data=['checking_account','duration','credit_hist','purpose',
                                                                       'credit_amount','savings','employment_since','installment_rate',
                                                                 'sex_status','debtors','residence_since','property',
                                                                       'age','installment_other','housing','existing_credits',
                                                                       'job','liable_people','tel','foreign']))
        # ['checking_account','duration','credit_hist','purpose',
        #                                                         'credit_amount','savings','employment_since','installment_rate',
        #                                                         'sex_status','debtors','residence_since','property','age',
        #                                                         'installment_other','housing','existing_credits','job',
        #                                                         'liable_people','tel','foreign']
        if suspect != true_recipient:
            print('ERROR: Detection is not working properly.')
            print('Removing: ' + str(file_name_full))
            os.remove(file_name_full)


def test_specific_file():
    file_name_full = 'parameter_guidelines/fingerprinted_data/german_credit/attr_subset_16/universal_g2.5_x1_l8_u1_sk0.csv'
    fp_data = pd.read_csv(file_name_full, low_memory=False)
    scheme = Universal(gamma=2.5, xi=1, fingerprint_bit_length=8, number_of_recipients=100)

    suspect = scheme.detection(fp_data, secret_key=0, target_attribute='target', primary_key_attribute='Id',
                                   original_attributes=pd.Series(
                                       data=['checking_account', 'duration', 'credit_hist', 'purpose',
                                             'credit_amount', 'savings', 'employment_since', 'installment_rate',
                                             'sex_status', 'debtors', 'residence_since', 'property',
                                             'age', 'installment_other', 'housing', 'existing_credits',
                                             'job', 'liable_people', 'tel', 'foreign']))
    print(suspect)


def main():
    nursery()


if __name__ == '__main__':
    #main()
    #test_detection()
    test_specific_file()
