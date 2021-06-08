from utilities import *
from scheme import Universal
import os.path
import glob


def main():
    data = pd.read_csv('datasets/adult.csv', na_values='?')
    data = data.dropna(axis=0).reset_index().drop('index', axis=1)

    gammae = [1,2,3,4,5,6, 10,12,15,18,20,25,30,35,40,50,60,70,80,100,200]
    fplen = 32
    numbuyers = 100
    target = 'income'
    exclude = [target]
    for g in gammae:
        for sk in range(100):
            file_string = 'parameter_guidelines/fingerprinted_data/adult/universal_g{}_x{}_l{}_u{}_sk{}.csv'.format(g, 1, fplen, 1, sk)
            if not os.path.isfile(file_string):
                scheme = Universal(gamma=g, xi=1, fingerprint_bit_length=fplen, number_of_recipients=numbuyers)
                fingerprinted_data = scheme.insertion(data, recipient_id=1, secret_key=sk,
                                                      exclude=exclude, write_to=file_string)
                #fingerprinted_data.get_dataframe().to_csv(file_string, index=False)
            else:
                print('File already exists; skipping gamma={}, no.{}'.format(g, sk))


def test_detection():
    directoryPath = os.path.join("parameter_guidelines", "fingerprinted_data", "adult")
    for file_name_full in glob.glob(directoryPath + '/*.csv'):
        file_name = file_name_full.split("\\")[-1].split('_')
        gamma = int(file_name[1][1:])
        sk = int(file_name[-1][2:-4])
        true_recipient = int(file_name[4][1:])
        fp_len = int(file_name[3][1:])
        xi = int(file_name[2][1:])

        fp_data = pd.read_csv(file_name_full, low_memory=False)
        scheme = Universal(gamma=gamma, xi=xi, fingerprint_bit_length=fp_len, number_of_recipients=100)

        suspect = scheme.detection(fp_data, secret_key=sk, exclude=['income'])
        if suspect != true_recipient:
            print('ERROR: Detection is not working properly.')
            print('Removing: ' + str(file_name_full))
            os.remove(file_name_full)


if __name__ == '__main__':
    main()
