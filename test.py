import unittest
from schemes.ak_scheme.ak_scheme import AKScheme
from schemes.block_scheme.block_scheme import BlockScheme
from schemes.two_level_scheme.two_level_scheme import TwoLevelScheme
from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood
from attacks.horizontal_subset_attack import SubsetAttack
from utils import *


class TestUtils(unittest.TestCase):
    def test_import_fingerprinted_dataset(self):
        """
        Test that the function imports something
        """
        data = ["AK", "covtype_data_int_sample", 10, 1, 0]
        result = import_fingerprinted_dataset(data[0], data[1], data[2], data[3], data[4])
        self.assertIsNotNone(result)

    def test_set_bit_no_change(self):
        """
        Test that the function sets the right value for a given positive integer
        in case when the output should remain the same.
        """
        data = [3, 1, 1]
        result = set_bit(data[0], data[1], data[2])
        self.assertEqual(result, data[0])

    def test_set_bit_change(self):
        """
        Test that the function sets the right value for a given positive integer
        in case when the output should be different than the input.
        """
        data = [3, 1, 0]
        result = set_bit(data[0], data[1], data[2])
        self.assertEqual(result, 1)

    def test_set_bit_negative(self):
        """
        Test the output of the set_bit function when the input is negative integer
        """
        data = [-3, 1, 0]
        result = set_bit(data[0], data[1], data[2])
        self.assertEqual(result, -1)


class TestAKScheme(unittest.TestCase):
    def test_detection_unmodified_small_dataset(self):
        """
        Test if the insertion algorithm finds the correct suspicious buyer after the
        insertion algorithm with the same parameters.
        """
        data = [5, 1, 16, 10, 333, "covtype_data_int_sample", 0]
        scheme = AKScheme(gamma=data[0], xi=data[1], fingerprint_bit_length=data[2], number_of_buyers=data[3],
                          secret_key=data[4])
        scheme.insertion(dataset_name=data[5], buyer_id=data[6])
        result = scheme.detection("covtype_data_int_sample", real_buyer_id=data[6])
        self.assertEqual(result, 0)

    def test_detection_unmodified(self):
        """
        Test if the insertion algorithm finds the correct suspicious buyer after the
        insertion algorithm with the same parameters.
        """
        data = [10, 1, 96, 10, 333, "covtype_data_int", 0]
        scheme = AKScheme(gamma=data[0], xi=data[1], fingerprint_bit_length=data[2], number_of_buyers=data[3],
                          secret_key=data[4])
        scheme.insertion(dataset_name=data[5], buyer_id=data[6])
        result = scheme.detection("covtype_data_int", real_buyer_id=data[6])
        self.assertEqual(result, 0)


class TestBlockScheme(unittest.TestCase):
    def test_insertion_algorithm(self):
        scheme = BlockScheme(beta=4, xi=2, fingerprint_bit_length=16, number_of_buyers=10, secret_key=333)
        result = scheme.insertion(dataset_name="covtype_data_int_sample", buyer_id=0)
        self.assertTrue(result)

    def test_detection_algorithm(self):
        scheme = BlockScheme(beta=4, xi=2, fingerprint_bit_length=16, number_of_buyers=10, secret_key=333)
        result = scheme.detection(dataset_name="covtype_data_int_sample", real_buyer_id=0)
        self.assertEqual(result, 0)

    def test_insertion_detection_big_dataset(self):
        scheme = BlockScheme(beta=7, xi=1, fingerprint_bit_length=96, number_of_buyers=10, secret_key=333)
        scheme.insertion(dataset_name="covtype_data_int", buyer_id=0)
        result = scheme.detection(dataset_name="covtype_data_int", real_buyer_id=0)
        self.assertEqual(result, 0)


class TestTwoLevelScheme(unittest.TestCase):
    def test_insertion_algorithm(self):
        scheme = TwoLevelScheme(gamma_1=5, gamma_2=5, xi=2, alpha_1=0.01, alpha_2=0.01, alpha_3=0.01,
                                fingerprint_bit_length=16, number_of_buyers=10, secret_key=333)
        result = scheme.insertion(dataset_name="covtype_data_int_sample", buyer_id=0)
        self.assertTrue(result)

    def test_detection_algorithm(self):
        scheme = TwoLevelScheme(gamma_1=5, gamma_2=5, xi=2, alpha_1=0.01, alpha_2=0.01, alpha_3=0.01,
                                fingerprint_bit_length=16, number_of_buyers=10, secret_key=333)
        result = scheme.detection(dataset_name="covtype_data_int_sample", real_buyer_id=0)
        self.assertEqual(result, 0)

    def test_scheme_big_dataset(self):
        scheme = TwoLevelScheme(gamma_1=5, gamma_2=5, xi=2, alpha_1=0.01, alpha_2=0.01, alpha_3=0.01,
                                fingerprint_bit_length=16, number_of_buyers=10, secret_key=333)
        scheme.insertion(dataset_name="covtype_data_int", buyer_id=1)
        result = scheme.detection(dataset_name="covtype_data_int", real_buyer_id=1)
        self.assertEqual(result, 1)


class TestCategoricalNeighbourhood(unittest.TestCase):
    def test_scheme_default_init(self):
        scheme = CategoricalNeighbourhood(gamma=10, xi=2, fingerprint_bit_length=16, number_of_buyers=10,
                                          secret_key=333)
        result_1 = scheme.distance_based
        result_2 = scheme.k
        result = not result_1 and result_2 == 10
        self.assertTrue(result)

    def test_scheme_init(self):
        scheme = CategoricalNeighbourhood(gamma=10, xi=2, fingerprint_bit_length=16, number_of_buyers=10,
                                          secret_key=333, distance_based=True, d=1)
        result_1 = scheme.distance_based
        result_2 = scheme.d
        result = result_1 and result_2 == 1
        self.assertTrue(result)

    def test_insertion(self):
        scheme = CategoricalNeighbourhood(gamma=10, xi=2, fingerprint_bit_length=32, number_of_buyers=10,
                                          secret_key=333)
        result = scheme.insertion(dataset_name="german_credit", buyer_id=2)
        self.assertTrue(result)

    def test_detection(self):
        scheme = CategoricalNeighbourhood(gamma=10, xi=2, fingerprint_bit_length=32, number_of_buyers=10,
                                          secret_key=333)
        result = scheme.detection(dataset_name="german_credit", real_buyer_id=2)
        self.assertEqual(result, 2)


class TestAttacks(unittest.TestCase):
    def test_subset_attack(self):
        attack = SubsetAttack()
        dataset, primary_key = import_dataset("german_credit")
        frac = 0.95
        result = attack.run(dataset=dataset, fraction=frac)
        self.assertEqual(len(result), frac*len(dataset))

    def test_small_subset_attack(self):
        attack = SubsetAttack()
        dataset, primary_key = import_dataset("german_credit")
        frac = 0.1
        result = attack.run(dataset=dataset, fraction=frac)
        self.assertEqual(len(result), frac*len(dataset))


if __name__ == '__main__':
    unittest.main()
