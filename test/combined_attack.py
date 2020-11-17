from attacks.combined_attack import CombinedAttack
from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood
import pandas as pd
from time import time


def test_func_and_time():
    data = pd.read_csv('../datasets/nursery.csv')
    attack = CombinedAttack()

    # start the run
    start = time()
    altered = attack.run(data, 0.1, 1, 0.9)  # remaining portion, deleted absolute, flipped portion
    finish = time()
    print('Breast Cancer dataset')
    print(data)
    print('Altered: params(0.4, 3, 0.2)')
    print(altered)
    print('Time: ' + str(finish - start) + " sec.")


def test_fp_and_attack():
    scheme = CategoricalNeighbourhood(gamma=5, xi=1, fingerprint_bit_length=8)
    start = time()
    fingerprinted = scheme.insertion(dataset_name='german_credit', buyer_id=0, secret_key=111)
    finish_insertion = time()
    print('Insertion time: ' + str(finish_insertion - start))
    attack = CombinedAttack()
    start_attack = time()
    altered = attack.run(fingerprinted, 0.5, 1, 0.3)  # remaining potion, deleted absolute, flipped portion
    finish_attack = time()
    print('Attack time: ' + str(finish_attack-start_attack))
    start_detection = time()
    suspect = scheme.detection(dataset_name='german_credit', real_buyer_id=0, secret_key=111, dataset=altered)
    finish = time()
    print('Detection time: ' + str(finish - start_detection))
    print(suspect)
    print('Full experiment time: ' + str(finish - start) + " sec.")


if __name__ == '__main__':
    test_fp_and_attack()
