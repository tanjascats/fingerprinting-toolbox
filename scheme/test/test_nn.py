import unittest

from scheme import CategoricalNeighbourhood


class MyTestCase(unittest.TestCase):
    def test_init(self):
        scheme = CategoricalNeighbourhood(gamma=2)
        self.assertIsNotNone(scheme)

    def test_insertion(self):
        scheme = CategoricalNeighbourhood(gamma=2)
        secret_key = 123
        recipient = 0
        data = '../../datasets/breast_cancer_full.csv'
        fingerprinted = scheme.insertion(data, recipient, secret_key)
        self.assertIsNotNone(fingerprinted)

    def test_detection(self):
        scheme = CategoricalNeighbourhood(gamma=2)
        secret_key = 123
        recipient = 0
        data = '../../datasets/breast_cancer_full.csv'
        fingerprinted = scheme.insertion(data, recipient, secret_key)
        suspect = scheme.detection(fingerprinted, secret_key, data)
        self.assertEqual(suspect, recipient)


if __name__ == '__main__':
    unittest.main()
