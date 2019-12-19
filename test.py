import unittest
from schemes.ak_scheme.ak_scheme import AKScheme
from schemes.block_scheme.block_scheme import BlockScheme
from schemes.two_level_scheme.two_level_scheme import TwoLevelScheme
from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood
from attacks.horizontal_subset_attack import HorizontalSubsetAttack
from attacks.bit_flipping_attack import BitFlippingAttack
from attacks.vertical_subset_attack import VerticalSubsetAttack
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
        scheme = CategoricalNeighbourhood(gamma=20, xi=2, fingerprint_bit_length=16, number_of_buyers=10,
                                          secret_key=333)
        result = scheme.insertion(dataset_name="german_credit", buyer_id=1)
        self.assertIsNotNone(result)

    def test_detection(self):
        scheme = CategoricalNeighbourhood(gamma=20, xi=2, fingerprint_bit_length=16, number_of_buyers=10,
                                          secret_key=333)
        result = scheme.detection(dataset_name="german_credit", real_buyer_id=1)
        self.assertEqual(result, 1)

    """
    testing on Breast Cancer data with 286 instances
    """
    def test_scheme_breast_cancer(self):
        scheme = CategoricalNeighbourhood(gamma=5, xi=2, fingerprint_bit_length=16, number_of_buyers=10,
                                          secret_key=333)
        scheme.insertion(dataset_name="breast_cancer", buyer_id=2)
        result = scheme.detection(dataset_name="breast_cancer", real_buyer_id=2)
        self.assertEqual(result, 2)

    """
    testing on Breast Cancer data with 286 instances
    """
    def test_false_scheme_breast_cancer(self):
        scheme = CategoricalNeighbourhood(gamma=7, xi=2, fingerprint_bit_length=16, number_of_buyers=10,
                                          secret_key=333)
        scheme.insertion(dataset_name="breast_cancer", buyer_id=0)
        result = scheme.detection(dataset_name="breast_cancer", real_buyer_id=0)
        self.assertNotEqual(result, 2)


class TestBlindCategoricalNeighbScheme(unittest.TestCase):
    def test_insertion(self):
        scheme = CategoricalNeighbourhood(gamma=2, xi=2, fingerprint_bit_length=16, number_of_buyers=10,
                                          secret_key=333)
        result = scheme.blind_insertion(dataset_name="breast_cancer", buyer_id=0)
        self.assertIsNotNone(result)

    def test_detection(self):
        scheme = CategoricalNeighbourhood(gamma=1, xi=2, fingerprint_bit_length=16, number_of_buyers=10,
                                          secret_key=333)
        scheme.blind_insertion(dataset_name="breast_cancer", buyer_id=5)
        result = scheme.blind_detection(dataset_name="breast_cancer", real_buyer_id=5)
        self.assertEqual(result, 5)

    def test_detection_2(self):
        scheme = CategoricalNeighbourhood(gamma=5, xi=2, fingerprint_bit_length=16, number_of_buyers=10,
                                          secret_key=333)
        scheme.blind_insertion(dataset_name="breast_cancer", buyer_id=5)
        result = scheme.blind_detection(dataset_name="breast_cancer", real_buyer_id=5)
        self.assertEqual(result, 5)


class TestAttacks(unittest.TestCase):
    def test_subset_attack(self):
        attack = HorizontalSubsetAttack()
        dataset, primary_key = import_dataset("german_credit")
        frac = 0.95
        result = attack.run(dataset=dataset, fraction=frac)
        self.assertEqual(len(result), frac*len(dataset))

    def test_small_subset_attack(self):
        attack = HorizontalSubsetAttack()
        dataset, primary_key = import_dataset("german_credit")
        frac = 0.1
        result = attack.run(dataset=dataset, fraction=frac)
        self.assertEqual(len(result), frac*len(dataset))

    def test_zero_bit_flipping_attack(self):
        attack = BitFlippingAttack()
        dataset, primary_key = import_dataset("german_credit")
        frac = 0
        result = attack.run(dataset=dataset, fraction=frac)
        test_column = dataset.columns.tolist()[1]
        self.assertEqual(result[test_column][:].tolist(), result[test_column][:].tolist())

    def test_bit_flipping_attack(self):
        attack = BitFlippingAttack()
        dataset, primary_key = import_dataset("german_credit")
        frac = 0.5
        result = attack.run(dataset=dataset, fraction=frac)
        self.assertEqual(result["property"][568], 'A122')

    def test_bit_flipping_attack_1(self):
        attack = BitFlippingAttack()
        dataset, primary_key = import_dataset("german_credit")
        frac = 0.5
        result = attack.run(dataset=dataset, fraction=frac)
        self.assertNotEqual(result['property'][568], dataset['property'][568])

    def test_zero_vertical_subset_attack(self):
        attack = VerticalSubsetAttack()
        dataset, primary_key = import_dataset("german_credit")
        subset = 0
        result = attack.run_random(dataset=dataset, number_of_columns=subset)
        self.assertEqual(len(result.columns), len(dataset.columns))

    def test_vertical_subset_attack(self):
        attack = VerticalSubsetAttack()
        dataset, primary_key = import_dataset("german_credit")
        subset = 2
        result = attack.run_random(dataset=dataset, number_of_columns=subset)
        self.assertEqual(len(result.columns), len(dataset.columns)-2)


if __name__ == '__main__':
    unittest.main()
