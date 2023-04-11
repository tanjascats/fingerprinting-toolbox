import pandas as pd
import bitstring
import unittest

from scheme import AKScheme
from datasets import *


class TestScheme(unittest.TestCase):
    def test_create_fingerprint(self):
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

    def test_no_target_datapath(self):
        scheme = AKScheme(gamma=1)
        secret_key = 123
        recipient = 0
        data = '../../datasets/breast_cancer_wisconsin.csv'
        fingerprinted_data = scheme.insertion(data, recipient, secret_key)
        suspect = scheme.detection(fingerprinted_data, secret_key)
        self.assertEqual(recipient, suspect)

    def test_exclude_datapath(self):
        scheme = AKScheme(gamma=1)
        secret_key = 123
        recipient = 0
        data = '../../datasets/breast_cancer_wisconsin.csv'
        dataframe = pd.read_csv(data)
        fingerprinted_data = scheme.insertion(data, recipient, secret_key, exclude=['clump-thickness'])
        # suspect = scheme.detection(fingerprinted_data, secret_key, exclude=['clump-thickness'])
        self.assertTrue(fingerprinted_data.dataframe['clump-thickness'].equals(dataframe['clump-thickness']))

    def test_exclude_pandasframe(self):
        scheme = AKScheme(gamma=1)
        secret_key = 123
        recipient = 0
        data = '../../datasets/breast_cancer_wisconsin.csv'
        dataframe = pd.read_csv(data)
        fingerprinted_data = scheme.insertion(dataframe, recipient, secret_key, exclude=['clump-thickness'])
        # suspect = scheme.detection(fingerprinted_data, secret_key, exclude=['clump-thickness'])
        self.assertTrue(fingerprinted_data.dataframe['clump-thickness'].equals(dataframe['clump-thickness']))

    def test_exclude(self):
        scheme = AKScheme(gamma=1)
        secret_key = 123
        recipient = 0
        data = '../../datasets/breast_cancer_wisconsin.csv'
        dataframe = pd.read_csv(data)
        fingerprinted_data = scheme.insertion(dataframe, recipient, secret_key, exclude=['clump-thickness'])
        # suspect = scheme.detection(fingerprinted_data, secret_key, exclude=['clump-thickness'])
        self.assertFalse(fingerprinted_data.dataframe['uniformity-of-cell-size'].equals(
            dataframe['uniformity-of-cell-size']))

    def test_exclude_detection(self):
        scheme = AKScheme(gamma=1)
        secret_key = 123
        recipient = 0
        data = '../../datasets/breast_cancer_wisconsin.csv'
        dataframe = pd.read_csv(data)
        exclude = ['clump-thickness']
#                   'uniformity-of-cell-shape',
#                   'marginal-adhesion']
#                   'single-epithelial-cell-size',
#                   'bare-nuclei',
#                   'bland-chromatin']
#                   'sample-code-number',
#                   'normal-nucleoli']
        fingerprinted_data = scheme.insertion(dataframe, recipient, secret_key, exclude=exclude)
        suspect = scheme.detection(fingerprinted_data, secret_key, exclude=exclude)
        self.assertEqual(recipient, suspect)

    def test_exclude_multi(self):
        scheme = AKScheme(gamma=1)
        secret_key = 123
        recipient = 0
        data = '../../datasets/breast_cancer_wisconsin.csv'
        dataframe = pd.read_csv(data)
        exclude = ['clump-thickness',
                   'uniformity-of-cell-shape',
                   'marginal-adhesion',
                   'single-epithelial-cell-size',
                   'bare-nuclei',
                   'bland-chromatin',
                   'sample-code-number',
                   'normal-nucleoli']
        fingerprinted_data = scheme.insertion(dataframe, recipient, secret_key, exclude=exclude)
        suspect = scheme.detection(fingerprinted_data, secret_key, exclude=exclude)
        self.assertEqual(recipient, suspect)

    def test_exclude_multi_2(self):
        scheme = AKScheme(gamma=1)
        secret_key = 123
        recipient = 0
        data = '../../datasets/breast_cancer_wisconsin.csv'
        dataframe = pd.read_csv(data)
        exclude = ['clump-thickness',
                   'uniformity-of-cell-shape',
                   'marginal-adhesion',
                   'single-epithelial-cell-size',
                   'bare-nuclei',
                   'bland-chromatin',
                   'sample-code-number',
                   'normal-nucleoli']
        fingerprinted_data = scheme.insertion(dataframe, recipient, secret_key, exclude=exclude)
        # suspect = scheme.detection(fingerprinted_data, secret_key, exclude=exclude)
        self.assertTrue(dataframe['clump-thickness'].equals(fingerprinted_data.dataframe['clump-thickness']))

    def test_exclude_multi_3(self):
        scheme = AKScheme(gamma=1)
        secret_key = 123
        recipient = 0
        data = '../../datasets/breast_cancer_wisconsin.csv'
        dataframe = pd.read_csv(data)
        exclude = ['clump-thickness',
                   'uniformity-of-cell-shape',
                   'marginal-adhesion',
                   'single-epithelial-cell-size',
                   'bare-nuclei',
                   'bland-chromatin',
                   'sample-code-number',
                   'normal-nucleoli']
        fingerprinted_data = scheme.insertion(dataframe, recipient, secret_key, exclude=exclude)
        # suspect = scheme.detection(fingerprinted_data, secret_key, exclude=exclude)
        self.assertFalse(dataframe['mitoses'].equals(fingerprinted_data.dataframe['mitoses']))

    def test_include(self):
        scheme = AKScheme(gamma=1)
        secret_key = 123
        recipient = 0
        data = '../../datasets/breast_cancer_wisconsin.csv'
        dataframe = pd.read_csv(data)
        fingerprinted_data = scheme.insertion(dataset=dataframe, recipient_id=recipient, secret_key=secret_key,
                                              include=['clump-thickness'])
        suspect = scheme.detection(fingerprinted_data, secret_key, include=['clump-thickness'])
        self.assertTrue(fingerprinted_data.dataframe['uniformity-of-cell-size'].equals(
            dataframe['uniformity-of-cell-size']))

    def test_include_1(self):
        scheme = AKScheme(gamma=1)
        secret_key = 123
        recipient = 0
        data = '../../datasets/breast_cancer_wisconsin.csv'
        dataframe = pd.read_csv(data)
        fingerprinted_data = scheme.insertion(dataset=dataframe, recipient_id=recipient, secret_key=secret_key,
                                              include=['clump-thickness'])
        suspect = scheme.detection(fingerprinted_data, secret_key, include=['clump-thickness'])
        self.assertFalse(fingerprinted_data.dataframe['clump-thickness'].equals(
            dataframe['clump-thickness']))

    def test_include_2(self):
        scheme = AKScheme(gamma=1)
        secret_key = 123
        recipient = 0
        data = '../../datasets/breast_cancer_wisconsin.csv'
        dataframe = pd.read_csv(data)
        fingerprinted_data = scheme.insertion(dataset=dataframe, recipient_id=recipient, secret_key=secret_key,
                                              include=['clump-thickness'])
        suspect = scheme.detection(fingerprinted_data, secret_key, include=['clump-thickness'])
        self.assertEqual(recipient, suspect)

    def test_include_multi(self):
        scheme = AKScheme(gamma=1)
        secret_key = 123
        recipient = 0
        data = '../../datasets/breast_cancer_wisconsin.csv'
        dataframe = pd.read_csv(data)
        include = ['clump-thickness', 'mitoses']
        fingerprinted_data = scheme.insertion(dataset=dataframe, recipient_id=recipient, secret_key=secret_key,
                                              include=include)
        suspect = scheme.detection(fingerprinted_data, secret_key, include=include)
        self.assertEqual(recipient, suspect)

    def test_include_fail(self):
        scheme = AKScheme(gamma=1)
        secret_key = 123
        recipient = 0
        data = '../../datasets/breast_cancer_wisconsin.csv'
        dataframe = pd.read_csv(data)
        include = ['clump-thickness', 'mitoses']
        fingerprinted_data = scheme.insertion(dataset=dataframe, recipient_id=recipient, secret_key=secret_key,
                                              include=include)
        suspect = scheme.detection(fingerprinted_data, secret_key)
        self.assertNotEqual(recipient, suspect)

