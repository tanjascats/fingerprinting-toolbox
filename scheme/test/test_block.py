import unittest
import os

from scheme import BlockScheme


class TestBlockScheme(unittest.TestCase):
    def test_init(self):
        beta = 3
        scheme = BlockScheme(beta=beta)
        self.assertEquals(scheme.beta, beta)

    def test_insertion_integer_data(self):
        scheme = BlockScheme(beta=2)
        secret_key = 123
        recipient = 0
        data = '../../datasets/breast_cancer_wisconsin.csv'
        fingerprinted = scheme.insertion(data, recipient, secret_key)
        # suspect = scheme.detection(fingerprinted, secret_key)
        self.assertIsNotNone(fingerprinted)

    def test_detection_integer_data(self):
        scheme = BlockScheme(beta=2)
        secret_key = 123
        recipient = 0
        data = '../../datasets/breast_cancer_wisconsin.csv'
        fingerprinted = scheme.insertion(data, recipient, secret_key)
        suspect = scheme.detection(fingerprinted, secret_key)
        self.assertEquals(suspect, recipient)


