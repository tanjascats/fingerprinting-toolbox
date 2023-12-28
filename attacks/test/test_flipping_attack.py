import pandas as pd
import unittest
import random
from time import time

from scheme import Universal
from attacks import FlippingAttack
from datasets import *


class TestFlipping(unittest.TestCase):
    def test_runtime_weak_under_min(self):
        start = time()
        attack = FlippingAttack()
        recipient = 4
        fingerprinted_data = pd.read_csv('evaluation/fingerprinted_data/breast_cancer_w/'
                                         'breast_cancer_w_l32_g1.11_x1_4370315727_4.csv')

        attacked_data = attack.run(fingerprinted_data, 0.05, random_state=1,
                                   keep_columns=['class'])
        runtime_sec = round(time()-start, 2)
        self.assertLessEqual(runtime_sec, 60)  # 3,2 sec approx

    def test_runtime_mid_attack_under_min(self):
        start = time()
        attack = FlippingAttack()
        recipient = 1
        fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/adult/'
                                         'universal_g1_x1_l32_u{}_sk0.csv'.format(recipient))

        attacked_data = attack.run(fingerprinted_data, 0.20, random_state=1,
                                   keep_columns=['income'])
        runtime_sec = round(time() - start, 2)
        self.assertLessEqual(runtime_sec, 60)  # {0.10:12sec, 0.15:26sec, 0.20:46sec, 25:64sec, 50:263sec}

    def test_runtime_strong_under_5min(self):
        start = time()
        attack = FlippingAttack()
        recipient = 1
        fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/adult/'
                                         'universal_g1_x1_l32_u{}_sk0.csv'.format(recipient))

        attacked_data = attack.run(fingerprinted_data, 0.5, random_state=1,
                                   keep_columns=['income'])
        runtime_sec = round(time() - start, 2)
        self.assertLessEqual(runtime_sec, 300)

    def test_flipping_more_LSBs(self):
        start = time()
        attack = FlippingAttack()
        recipient = 4
        fingerprinted_data = pd.read_csv('evaluation/fingerprinted_data/breast_cancer_w/'
                                         'breast_cancer_w_l32_g1.11_x1_4370315727_4.csv')

        attacked_data = attack.run(fingerprinted_data, 0.05, random_state=1, xi=3)
        self.assertIsNotNone(attacked_data)
