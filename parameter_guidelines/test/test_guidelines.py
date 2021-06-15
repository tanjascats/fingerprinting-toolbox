import unittest
from parameter_guidelines.guidelines import *


class TestGuidelines(unittest.TestCase):
    def test_fp_utility_run(self):
        data = Adult()
        target = 'income'
        user = 1
        n_folds = 5
        n_exp = 2
        utility = fingerprint_utility_knn(data, target, user, n_folds, n_exp)
        self.assertIsNotNone(utility)

    def test_fp_utility_results(self):
        data = Adult()
        target = 'income'
        user = 1
        n_folds = 5
        n_exp = 2
        utility = fingerprint_utility_knn(data, target, user, n_folds, n_exp)
        print(utility)
        self.assertEqual(len(utility), n_exp)

    def test_inverse_robustness(self):
        scheme = Universal(gamma=1, xi=1, fingerprint_bit_length=32, number_of_recipients=100)
        data = Adult()
        target = 'income'
        attack_granularity = 0.1
        n_exp = 100
        conf = 0.95
        attack = HorizontalSubsetAttack()
        inverse_rob_results = inverse_robustness(attack, scheme, data, exclude=[target],
                                                 attack_granularity=attack_granularity,
                                                 n_experiments=n_exp,
                                                 confidence_rate=conf)
        print(inverse_rob_results)
        self.assertIsNotNone(inverse_rob_results)

    def test_eval_fp_data(self):
        eval_data_path = 'parameter_guidelines/fingerprinted_data/adult/universal_g1_x1' \
                         '_l32_u1_sk0.csv'
        fp_data = pd.read_csv(eval_data_path)
        print(fp_data)
        scheme = Universal(gamma=1, fingerprint_bit_length=32, number_of_recipients=100, xi=1)
        suspect = scheme.detection(fp_data, secret_key=0, exclude='income')
        self.assertEqual(suspect, 1)

    def test_inverse_robustness_vertical_adult(self):
        attack = VerticalSubsetAttack()
        scheme = Universal(gamma=1, xi=1, fingerprint_bit_length=32, number_of_recipients=100)
        data = Adult().preprocessed()
        target = 'income'
        attack_granularity = 0.05
        n_experiments = 100
        confidence_rate = 0.9
        remaining = inverse_robustness(attack, scheme, data, exclude=[target],
                                       attack_granularity=attack_granularity,
                                       n_experiments=n_experiments,
                                       confidence_rate=confidence_rate)
        print(remaining)
        self.assertIsNotNone(remaining)
