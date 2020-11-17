import pandas as pd
import unittest

from scheme import NBNNScheme


class TestNBNNScheme(unittest.TestCase):
    def test_insertion(self):
        """Test that insertion changes the fingerprinted data set"""
        # todo: the algorithm fails for <2 attributes
        data = pd.DataFrame({'Id': [i for i in range(6)],
                             'Attr1': ['blue', 'red', 'blue', 'blue', 'yellow', 'red'],
                             'Attr2': ['yes', 'no', 'no', 'no', 'yes', 'yes']})
        scheme = NBNNScheme(gamma=1, secret_key=7, fingerprint_bit_length=8, k=3, number_of_buyers=3, xi=2)
        fingerprinted_data = scheme.insertion(data, 0)
        self.assertFalse(data.equals(fingerprinted_data))

    def test_detection(self):
        data = 'german_credit_full'  # fails for small (150 rows, 11 columns) datasets
        scheme = NBNNScheme(gamma=1, secret_key=7, fingerprint_bit_length=8, k=3, number_of_buyers=3, xi=2)
        fingerprinted_data = scheme.insertion(data, 0)

        suspect_id = scheme.detection(fingerprinted_data)
        print(suspect_id)
        self.assertEqual(0, suspect_id)
