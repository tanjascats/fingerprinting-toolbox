import unittest
from AK.AK import *


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
        data = [5, 1, 16, 10, 0, 333, "covtype_data_int_sample"]
        scheme = AK(gamma=data[0], xi=data[1], fingerprint_bit_length=data[2], number_of_buyers=data[3], buyer_id=data[4], secret_key=data[5])
        scheme.insertion(data[6])
        result = scheme.detection("covtype_data_int_sample")
        self.assertEqual(result, 0)

    def test_detection_unmodified(self):
        """
        Test if the insertion algorithm finds the correct suspicious buyer after the
        insertion algorithm with the same parameters.
        """
        data = [10, 1, 96, 10, 0, 333, "covtype_data_int"]
        scheme = AK(gamma=data[0], xi=data[1], fingerprint_bit_length=data[2], number_of_buyers=data[3],
                    buyer_id=data[4], secret_key=data[5])
        scheme.insertion(data[6])
        result = scheme.detection("covtype_data_int")
        self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()