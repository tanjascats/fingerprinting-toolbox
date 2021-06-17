from attacks._base import Attack
import time
import random
from datasets import Dataset


class HorizontalSubsetAttack(Attack):

    def __init__(self):
        super().__init__()

    """
    Runs the attack; gets a random subset of a dataset of size fraction*data_size
    fraction [0,1]
    """
    def run(self, dataset, strength=None, fraction=None, random_state=None):
        if isinstance(dataset, Dataset):
            dataset = dataset.get_dataframe()
        if strength is None and fraction is None:  # if strength is none and fraction also
            return None
        if strength is not None and fraction is not None:
            print('Both fraction and strength of horizontal attack are provided -- using fraction.')
        if strength is not None and fraction is None:
            fraction = 1.0 - strength
        if fraction < 0 or fraction > 1:
            return None

        start = time.time()
        subset = dataset.sample(frac=fraction, random_state=random_state)
        print("Subset attack runtime on " + str(int(fraction*len(dataset))) + " out of " + str(len(dataset)) +
              " entries: " + str(time.time()-start) + " sec.")
        return subset


class VerticalSubsetAttack(Attack):

    def __init__(self):
        super().__init__()

    """
    Runs the attack; gets a subset of columns without (a) predefined column(s)
    """
    def run(self, dataset, columns):
        start = time.time()
        if 'Id' in columns:
            print("Cannot delete Id columns of the data set!")
            subset = None
        else:
            subset = dataset.drop(columns, axis=1)
            print("Vertical subset attack runtime on " + str(len(columns)) + " out of " + str(len(dataset.columns)-1) +
                  " columns: " + str(time.time()-start) + " sec.")
        return subset

    """
    Runs the attack; gets a subset of columns without random 'number_of_columns' columns
    """
    def run_random(self, dataset, number_of_columns, seed, keep_columns=None):
        if number_of_columns >= len(dataset.columns)-1:
            print("Cannot delete all columns.")
            return None
        random.seed(seed)
        start = time.time()
        if 'Id' in dataset.columns:
            column_subset = random.sample(list(dataset.columns.drop(labels=["Id"])),
                                          k=number_of_columns)
        elif keep_columns is not None:
            column_subset = random.sample(list(dataset.columns.drop(labels=keep_columns)),
                                          k=number_of_columns)
        else:
            column_subset = random.sample(list(dataset.columns), k=number_of_columns)
        subset = dataset.drop(column_subset, axis=1)
        print(column_subset)

        print("Vertical subset attack runtime on " + str(number_of_columns) + " out of " + str(len(dataset.columns)-1) +
              " columns: " + str(time.time() - start) + " sec.")
        return subset
