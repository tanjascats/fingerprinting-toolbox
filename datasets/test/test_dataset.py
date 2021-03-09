import unittest
import pandas as pd

from .._dataset import Dataset


class TestDataset(unittest.TestCase):
    def test_creation_from_path(self):
        path = "datasets/breast_cancer_wisconsin.csv"
        dataset = Dataset(path=path, target_attribute='class', primary_key_attribute='sample-code-number')
        self.assertTrue(isinstance(dataset.dataframe, pd.DataFrame))

    def test_creation_from_dataframe(self):
        dataframe = pd.read_csv('datasets/breast_cancer_wisconsin.csv')
        dataset = Dataset(dataframe=dataframe, target_attribute='class', primary_key_attribute='sample-code-number')
        self.assertTrue(isinstance(dataset.dataframe, pd.DataFrame))

    def test_no_data_specified(self):
        self.assertRaises(ValueError, Dataset, 'class')

    def test_default_primary_key(self):
        path = "datasets/breast_cancer_wisconsin.csv"
        dataset = Dataset(path=path, target_attribute='class')
        self.assertTrue(isinstance(dataset.primary_key, pd.Series))

    def test_primary_key_type(self):
        path = "datasets/breast_cancer_wisconsin.csv"
        dataset = Dataset(path=path, target_attribute='class', primary_key_attribute='sample-code-number')
        self.assertTrue(isinstance(dataset.primary_key, pd.Series))

    def test_data_size(self):
        path = "datasets/breast_cancer_wisconsin.csv"
        dataset = Dataset(path=path, target_attribute='class', primary_key_attribute='sample-code-number')
        self.assertEqual(dataset.number_of_columns, 11)
