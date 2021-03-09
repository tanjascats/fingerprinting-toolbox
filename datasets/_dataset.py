from abc import ABC, abstractmethod
import pandas as pd


class Dataset(ABC):
    """
    Abstract scheme for structuring the dataset within this toolbox
    """
    # todo: default values
    def __init__(self, target_attribute, path=None, dataframe=None, primary_key_attribute=None):
        if path is None and dataframe is None:
            raise ValueError('Error defining a data set! Please provide either a path or a Pandas DataFrame to '
                             'instantiate a data set')

        self.path = path
        self.dataframe = dataframe

        if self.path is not None and not isinstance(self.path, str):
            raise TypeError('Data set path must be a string value.')
        elif self.path is not None:
            self.dataframe = pd.read_csv(self.path)

        if not isinstance(self.dataframe, pd.DataFrame):
            raise TypeError('Data frame must be type pandas.DataFrame')

        self._set_primary_key(primary_key_attribute)

        self.target_attribute = target_attribute
        if not isinstance(self.target_attribute, str):
            raise TypeError('Target attribute should be a string name of the attribute column')
        self.target = self.dataframe[self.target_attribute]

        self.columns = self.dataframe.columns
        self.number_of_rows, self.number_of_columns = self.dataframe.shape

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
