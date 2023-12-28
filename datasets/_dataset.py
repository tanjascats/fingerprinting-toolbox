from abc import ABC, abstractmethod
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder


class Dataset(ABC):
    """
    Abstract class used to represent the dataset within the toolbox
    """
    def __init__(self, target_attribute, path=None, dataframe=None, primary_key_attribute=None, na_values=None):
        if path is None and dataframe is None:
            raise ValueError('Error defining a data set! Please provide either a path or a Pandas DataFrame to '
                             'instantiate a data set')

        self.path = path
        self.dataframe = dataframe

        if self.path is not None and not isinstance(self.path, str):
            raise TypeError('Data set path must be a string value.')
        elif self.path is not None:
            self.dataframe = pd.read_csv(self.path, na_values=na_values)

        if not isinstance(self.dataframe, pd.DataFrame):
            raise TypeError('Data frame must be type pandas.DataFrame')

        self._set_primary_key(primary_key_attribute)

        self.set_target_attribute(target_attribute)

        self.columns = self.dataframe.columns
        self.number_of_rows, self.number_of_columns = self.dataframe.shape
        self._set_types()

        self.label_encoders = None

    def _set_types(self):
        self.categorical_attributes = self.dataframe.select_dtypes(include='object').columns
        self.decimal_attributes = self.dataframe.select_dtypes(include=['float64', 'float32'])
        self.integer_attributes = self.dataframe.select_dtypes(include=['int64', 'int32'])

    def _set_primary_key(self, primary_key_attribute):
        self.primary_key = None

        self.primary_key_attribute = primary_key_attribute
        if self.primary_key_attribute is None:
            # default primary key is the index
            self.primary_key = self.dataframe.index.to_series()

        # if the primary key attribute is specified
        elif not isinstance(self.primary_key_attribute, str):
            raise TypeError('Primary key attribute should be a string name of the attribute column')
        else:
            self.primary_key = self.dataframe[self.primary_key_attribute]

    def set_dataframe(self, new_dataframe):
        '''
        Sets the dataframe of the class instance to another dataframe
        :param new_dataframe: pandas.DataFrame instance
        :return: self
        '''
        if not isinstance(new_dataframe, pd.DataFrame):
            print('Dataset can only be set to a pandas.DataFrame instance')
            exit()
        self.dataframe = new_dataframe

        self.columns = self.dataframe.columns
        self.number_of_rows, self.number_of_columns = self.dataframe.shape
        return self

    def remove_primary_key(self):
        if self.primary_key_attribute is not None:
            self.set_dataframe(self.dataframe.drop(self.primary_key_attribute, axis=1))
        return self

    def remove_target(self):
        if self.target_attribute is not None:
            self.set_dataframe(self.dataframe.drop(self.target_attribute, axis=1))
        return self

    def remove_categorical(self):
        self.set_dataframe(self.dataframe.select_dtypes(exclude='object'))
        return self

    def add_column(self, name, values):
        self.dataframe[name] = values

        self.columns = self.dataframe.columns
        self.number_of_rows, self.number_of_columns = self.dataframe.shape
        return self

    def save(self, path):
        self.dataframe.to_csv(path, index=False)
        return self

    def get_target_attribute(self):
        return self.target_attribute

    def set_target_attribute(self, target_attribute):
        self.target_attribute = target_attribute
        if self.target_attribute is not None:
            if not isinstance(self.target_attribute, str):
                raise TypeError('Target attribute should be a string name of the attribute column')
            self.target = self.dataframe[self.target_attribute]
        else:
            self.target = None

    def get_target(self):
        return self.dataframe[self.target_attribute]

    def get_primary_key_attribute(self):
        return self.primary_key_attribute

    def get_dataframe(self):
        return self.dataframe.copy(deep=True)

    def get_features(self):
        non_features = []
        if self.target_attribute is not None:
            non_features.append(self.target_attribute)
        if self.primary_key_attribute is not None:
            non_features.append(self.primary_key_attribute)
        return self.dataframe.drop(non_features, axis=1)

    def clone(self):
        clone = Dataset(target_attribute=self.target_attribute, dataframe=self.get_dataframe(),
                        primary_key_attribute=self.get_primary_key_attribute())
        return clone

    def number_encode_categorical(self):
        relation = self.dataframe
        self.label_encoders = dict()
        for cat in self.categorical_attributes:
            label_enc = LabelEncoder()  # the current version of label encoder works in alphanumeric order
            relation[cat] = label_enc.fit_transform(relation[cat])
            self.label_encoders[cat] = label_enc
        self.set_dataframe(relation)
        return self

    def decode_categorical(self):
        relation = self.dataframe
        for cat in self.categorical_attributes:
            label_enc = self.label_encoders[cat]  # the current version of label encoder works in alphanumeric order
            relation[cat] = label_enc.inverse_transform(relation[cat])
        self.set_dataframe(relation)

    def get_distinct(self, attribute_index):
        attribute_name = self.columns[attribute_index]
        return self.dataframe[attribute_name].unique()

    def get_types(self):
        return self.dataframe.dtypes

    def dropna(self):
        '''
        Dropping samples (rows) with missing values. Returns the Dataset object.
        '''
        self.dataframe = self.dataframe.dropna()
        return self

    def drop(self, labels, axis=1):
        '''
        Equivalent of pandas drop
        '''
        if axis == 1:
            self.dataframe = self.dataframe.drop(labels, axis=1)
            self.columns = self.dataframe.columns
            self.number_of_rows, self.number_of_columns = self.dataframe.shape
            self._set_types()
        elif axis == 0:
            self.dataframe = self.dataframe.drop(labels, axis=0)
            self.number_of_rows, self.number_of_columns = self.dataframe.shape
        return self


class GermanCredit(Dataset):
    def __init__(self):
        path = os.path.dirname(os.path.realpath(__file__)) + '/german_credit_full.csv'
        super().__init__(path=path, target_attribute='target', primary_key_attribute='Id')


class BreastCancerWisconsin(Dataset):
    def __init__(self):
        path = os.path.dirname(os.path.realpath(__file__)) + '/breast_cancer_wisconsin.csv'
        super().__init__(path=path, target_attribute='class', primary_key_attribute='sample-code-number')


class Adult(Dataset):
    def __init__(self):
        path = os.path.dirname(os.path.realpath(__file__)) + '/adult.csv'
        super().__init__(path=path, target_attribute='income', na_values='?')


class BreastCancer(Dataset):
    def __init__(self):
        path = os.path.dirname(os.path.realpath(__file__)) + '/breast_cancer_full.csv'
        super().__init__(path=path, primary_key_attribute='Id', target_attribute='recurrence')


class Nursery(Dataset):
    def __init__(self):
        path = os.path.dirname(os.path.realpath(__file__)) + '/nursery_full.csv'
        super().__init__(path=path, primary_key_attribute='Id', target_attribute='target')


class Mushrooms(Dataset):
    def __init__(self):
        path = os.path.dirname(os.path.realpath(__file__)) + '/mushroom_full.csv'
        super().__init__(path=path, primary_key_attribute='Id', target_attribute='target')


class CovTypeNumeric(Dataset):
    def __init__(self):
        path = os.path.dirname(os.path.realpath(__file__)) + '/covtype_numeric.csv'
        super().__init__(path=path, primary_key_attribute='Id', target_attribute='Cover_Type')
