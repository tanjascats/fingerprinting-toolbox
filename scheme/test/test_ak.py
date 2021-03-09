import pandas as pd
import bitstring
import unittest

from scheme import AKScheme
from datasets import *


class TestScheme(unittest.TestCase):
    def test_base_fingerprint_creation(self):
        scheme = AKScheme(gamma=2)
        secret_key = 123
        recipient = 0
        fingerprint = scheme.create_fingerprint(recipient_id=recipient, secret_key=secret_key)
        self.assertIsNotNone(fingerprint)

    def test_base_fingerprint_detection(self):
        scheme = AKScheme(gamma=2)
        secret_key = 123
        recipient = 0
        fingerprint = scheme.create_fingerprint(recipient_id=recipient, secret_key=secret_key)
        suspect = scheme.detect_potential_traitor(fingerprint=fingerprint.bin, secret_key=secret_key)
        self.assertEqual(suspect, recipient)

    def test_base_fingerprint_detection_bitstring_input(self):
        scheme = AKScheme(gamma=2)
        secret_key = 123
        recipient = 0
        fingerprint = scheme.create_fingerprint(recipient_id=recipient, secret_key=secret_key)
        suspect = scheme.detect_potential_traitor(fingerprint=fingerprint, secret_key=secret_key)
        self.assertEqual(suspect, recipient)

    def test_base_wrong_suspect(self):
        scheme = AKScheme(gamma=2)
        secret_key = 123
        recipient = 0
        fingerprint = scheme.create_fingerprint(recipient_id=recipient, secret_key=secret_key)
        suspect = scheme.detect_potential_traitor(fingerprint=fingerprint, secret_key=secret_key+1)
        self.assertNotEqual(suspect, recipient)

    def test_base_no_suspect(self):
        scheme = AKScheme(gamma=2)
        secret_key = 123
        recipient = 0
        fingerprint = scheme.create_fingerprint(recipient_id=recipient, secret_key=secret_key)
        suspect = scheme.detect_potential_traitor(fingerprint=fingerprint, secret_key=secret_key + 1)
        self.assertEqual(suspect, -1)


class TestAKScheme(unittest.TestCase):
    def test_insertion_all_numerical(self):
        scheme = AKScheme(gamma=2)
        secret_key = 123
        recipient = 0
        data = BreastCancerWisconsin()
        fingerprinted = scheme.insertion(data, recipient, secret_key)
        self.assertListEqual(data.dataframe.columns.tolist(), fingerprinted.dataframe.columns.tolist())

    def test_fingerprinted_data_type(self):
        scheme = AKScheme(gamma=2)
        secret_key = 123
        recipient = 0
        data = BreastCancerWisconsin()
        fingerprinted = scheme.insertion(data, recipient, secret_key)
        self.assertIsInstance(fingerprinted, Dataset)

    def test_detection_all_numerical(self):
        scheme = AKScheme(gamma=2)
        secret_key = 123
        recipient = 0
        data = BreastCancerWisconsin()
        fingerprinted_data = scheme.insertion(data, recipient, secret_key)
        suspect = scheme.detection(fingerprinted_data, secret_key)
        self.assertEqual(recipient, suspect)
