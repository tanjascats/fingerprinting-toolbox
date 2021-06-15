import pandas as pd
import unittest
import random

from scheme import Universal
from attacks import VerticalSubsetAttack, HorizontalSubsetAttack
from datasets import *


class TestVerticalSubset(unittest.TestCase):
    def test_random_choice(self):
        data = Adult().preprocessed()
        scheme = Universal(gamma=1, xi=1, fingerprint_bit_length=32,
                           number_of_recipients=100)
        fingerprinted_data = scheme.insertion(data, recipient_id=1, secret_key=0,
                                              exclude=['income']).dataframe
        attack = VerticalSubsetAttack()

        attack_strength = 1
        attacked_data = attack.run_random(fingerprinted_data, attack_strength)
        print(attacked_data.columns)
        self.assertIsNotNone(attacked_data)

    def test_attack_detection(self):
        data = Adult().preprocessed()
        scheme = Universal(gamma=1, xi=1, fingerprint_bit_length=32,
                           number_of_recipients=100)
        recipient = 1
        fingerprinted_data = scheme.insertion(data, recipient_id=recipient, secret_key=0,
                                              exclude=['income']).dataframe
        attack = VerticalSubsetAttack()

        attack_strength = 1
        attacked_data = attack.run_random(fingerprinted_data, attack_strength, seed=0)
        suspect = scheme.detection(attacked_data, secret_key=0, exclude=['income'])
        self.assertEqual(suspect, recipient)

    def test_attack_from_file(self):
        scheme = Universal(gamma=1, xi=1, fingerprint_bit_length=32,
                           number_of_recipients=100)
        recipient = 1
        fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/adult/'
                                         'universal_g1_x1_l32_u{}_sk0.csv'.format(recipient))
        print(fingerprinted_data.columns)
        print(type(fingerprinted_data.columns))
        orig_attr = fingerprinted_data.columns.drop('income')
        attack = VerticalSubsetAttack()

        attack_strength = 1
        attacked_data = attack.run_random(fingerprinted_data, attack_strength, seed=0)
        suspect = scheme.detection(attacked_data, secret_key=0, exclude=['income'],
                                   original_attributes=orig_attr)
        self.assertEqual(suspect, recipient)

    def test_attack_from_file_2(self):
        scheme = Universal(gamma=1, xi=1, fingerprint_bit_length=32,
                           number_of_recipients=100)
        recipient = 1
        fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/adult/'
                                         'universal_g1_x1_l32_u{}_sk0.csv'.format(recipient))
        print(fingerprinted_data.columns)
        print(type(fingerprinted_data.columns))
        orig_attr = fingerprinted_data.columns.drop('income')
        attack = VerticalSubsetAttack()

        attack_strength = 4
        attacked_data = attack.run_random(fingerprinted_data, attack_strength, seed=0)
        suspect = scheme.detection(attacked_data, secret_key=0, exclude=['income'],
                                   original_attributes=orig_attr)
        self.assertEqual(suspect, recipient)

    def test_vertical_random_seed(self):
        scheme = Universal(gamma=1, xi=1, fingerprint_bit_length=32,
                           number_of_recipients=100)
        recipient = 1
        fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/adult/'
                                         'universal_g1_x1_l32_u{}_sk0.csv'.format(recipient))
        print(fingerprinted_data.columns)
        print(type(fingerprinted_data.columns))
        orig_attr = fingerprinted_data.columns.drop('income')
        attack = VerticalSubsetAttack()

        attack_strength = 4
        attacked_data = attack.run_random(fingerprinted_data, attack_strength, seed=1)
        suspect = scheme.detection(attacked_data, secret_key=0, exclude=['income'],
                                   original_attributes=orig_attr)

        attacked_data = attack.run_random(fingerprinted_data, attack_strength, seed=1)
        suspect = scheme.detection(attacked_data, secret_key=0, exclude=['income'],
                                   original_attributes=orig_attr)

    def test_vertical_delete_all(self):
        recipient = 1
        fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/adult/'
                                         'universal_g1_x1_l32_u{}_sk0.csv'.format(recipient))
        print(fingerprinted_data.columns)
        print(type(fingerprinted_data.columns))
        attack = VerticalSubsetAttack()

        attack_strength = 14
        attacked_data = attack.run_random(fingerprinted_data, attack_strength, seed=1)
        self.assertIsNone(attacked_data)
