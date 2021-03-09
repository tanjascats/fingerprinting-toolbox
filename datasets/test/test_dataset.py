import unittest
import pandas as pd

from .._dataset import Dataset, Adult


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

    def test_remove_primary_key(self):
        path = "datasets/breast_cancer_wisconsin.csv"
        primary_key_attribute = 'sample-code-number'
        dataset = Dataset(path=path, target_attribute='class', primary_key_attribute=primary_key_attribute)
        dataset.remove_primary_key()
        self.assertNotIn(primary_key_attribute, dataset.dataframe.columns)

    def test_remove_target(self):
        path = "datasets/breast_cancer_wisconsin.csv"
        target_attribute = 'class'
        dataset = Dataset(path=path, target_attribute=target_attribute, primary_key_attribute='sample-code-number')
        dataset.remove_target()
        self.assertNotIn(target_attribute, dataset.dataframe.columns)

    def test_target(self):
        path = "datasets/breast_cancer_wisconsin.csv"
        target_attribute = 'class'
        dataset = Dataset(path=path, target_attribute=target_attribute, primary_key_attribute='sample-code-number')
        self.assertIn(target_attribute, dataset.dataframe.columns)


class TestAdultDataset(unittest.TestCase):
    def test_primary_key(self):
        adult_data = Adult()
        self.assertIsNotNone(adult_data.primary_key)

    def test_size(self):
        adult_data = Adult()
        self.assertEqual(adult_data.number_of_rows, 48842)
