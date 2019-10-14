import unittest
from utils import *


class TestUtils(unittest.TestCase):
    def test_import_fingerprinted_dataset(self):
        """
        Test that the function imports something
        """
        data = ["AK", "covtype_data_int", 10, 1, 0]
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


if __name__ == '__main__':
    unittest.main()