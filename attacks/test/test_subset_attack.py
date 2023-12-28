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

    def test_repetitions(self):
        recipient = 1
        fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/adult/'
                                         'universal_g1_x1_l32_u{}_sk0.csv'.format(recipient))
        print(fingerprinted_data.columns)
        print(type(fingerprinted_data.columns))
        attack = VerticalSubsetAttack()

        attack_strength = 10
        attacked_data = attack.run_random(fingerprinted_data, attack_strength, seed=4)
        print(attacked_data.columns)
        self.assertEqual(len(attacked_data.columns), 15-attack_strength)

    def test_keep_columns(self):
        recipient = 1
        fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/adult/'
                                         'universal_g1_x1_l32_u{}_sk0.csv'.format(recipient))
        print(fingerprinted_data.columns)
        print(type(fingerprinted_data.columns))
        attack = VerticalSubsetAttack()

        attack_strength = 12
        keep = ['income']
        attacked_data = attack.run_random(fingerprinted_data, attack_strength, keep_columns=keep,
                                          seed=4)
        print(attacked_data.columns)
        self.assertIn('income', attacked_data.columns)


class TestHorizontalSubset(unittest.TestCase):
    def test_parameters_strength(self):
        attack = HorizontalSubsetAttack()
        recipient = 1
        fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/adult/'
                                         'universal_g1_x1_l32_u{}_sk0.csv'.format(recipient))
        attack_strength = 0.9
        attacked_data = attack.run(fingerprinted_data, strength=attack_strength)
        self.assertEqual(len(attacked_data), int(0.1*len(fingerprinted_data)))

    def test_parameters_fraction(self):
        attack = HorizontalSubsetAttack()
        recipient = 1
        fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/adult/'
                                         'universal_g1_x1_l32_u{}_sk0.csv'.format(recipient))
        fraction = 0.2
        attacked_data = attack.run(fingerprinted_data, fraction=fraction)
        self.assertEqual(len(attacked_data), int(0.2 * len(fingerprinted_data)))

    def test_parameters_strength_fraction(self):
        attack = HorizontalSubsetAttack()
        recipient = 1
        fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/adult/'
                                         'universal_g1_x1_l32_u{}_sk0.csv'.format(recipient))
        fraction = 0.2
        attack_strength = 0.9
        attacked_data = attack.run(fingerprinted_data, fraction=fraction, strength=attack_strength)
        self.assertEqual(len(attacked_data), int(0.2 * len(fingerprinted_data)))

    def test_parameters_none(self):
        attack = HorizontalSubsetAttack()
        recipient = 1
        fingerprinted_data = pd.read_csv('parameter_guidelines/fingerprinted_data/adult/'
                                         'universal_g1_x1_l32_u{}_sk0.csv'.format(recipient))

        attacked_data = attack.run(fingerprinted_data)
        self.assertIsNone(attacked_data)

    def test_false_miss_estimation(self):
        attack = HorizontalSubsetAttack()
        fingerprinted_data = pd.read_csv('evaluation/fingerprinted_data/breast_cancer_w/'
                                         'breast_cancer_w_l32_g1.11_x1_4370315727_4.csv')
        scheme = Universal(gamma=1, fingerprint_bit_length=32)
        fm = attack.false_miss_estimation(fingerprinted_data, 0.7, scheme)
        print(fm)
        self.assertLessEqual(fm, 1)
