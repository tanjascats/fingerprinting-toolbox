from attacks._base import Attack
import time
import random
from datasets import Dataset

class HorizontalSubsetAttack(Attack):

    def __init__(self):
        super().__init__()

    """
    Runs the attack; removes strength*data_size records
        strength [0,1]
    """
    def run(self, dataset, strength, random_state=None):
        if strength < 0 or strength > 1:
            return None

        start = time.time()
        fraction = 1.0 - strength
        if random_state is None:
            random_state = int(start)
        else:
            random_state = random_state
        subset = dataset.sample(frac=fraction, random_state=random_state)
        print("Subset attack runtime on removing " + str(int(strength*len(dataset))) + " out of " + str(len(dataset)) +
              " entries: " + str(time.time()-start) + " sec.")
        return subset

    def false_miss_estimation(self, dataset, strength, scheme):
        if isinstance(dataset, Dataset):
            data_len = dataset.number_of_rows
        else:
            data_len = len(dataset)
        fp_len = scheme.fingerprint_bit_length
        gamma = scheme.get_gamma()
        omega = round(data_len / (gamma * fp_len), 0)

        false_miss = 1.0 - (1.0 - strength**omega)**fp_len

        return false_miss


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
    def run_random(self, dataset, number_of_columns, target_attr=None, primary_key=None, random_state=None):
        if number_of_columns >= len(dataset.columns)-1:
            print("Cannot delete all columns.")
            return None

        start = time.time()
        exclude = []
        if target_attr is not None:
            exclude.append(target_attr)
        if primary_key is not None:
            exclude.append(primary_key)
        if random_state is None:
            random_state = int(start)
        else:
            random_state = random_state
        random.seed(random_state)
        column_subset = random.sample(dataset.columns.drop(labels=exclude).tolist(), k=number_of_columns)
        subset = dataset.drop(column_subset, axis=1)

        print("Vertical subset attack runtime on " + str(number_of_columns) + " out of " + str(len(dataset.columns)-1) +
              " columns.\n\tremoving: " + str(column_subset) + "\n\t" + str(time.time() - start) + " sec.")
        return subset

    def false_miss_estimation(self, dataset, scheme, strength_abs=None, strength_rel=None):
        if isinstance(dataset, Dataset):
            data_len = dataset.number_of_rows
            n_columns = dataset.number_of_columns
        else:
            data_len = len(dataset)
            n_columns = len(dataset.columns)
        if strength_rel is not None:
            strength = strength_rel
        elif strength_abs is not None:
            strength = strength_abs/n_columns
        fp_len = scheme.fingerprint_bit_length
        gamma = scheme.get_gamma()
        omega = round(data_len / (gamma * fp_len), 0)

        false_miss = 1.0 - (1.0 - strength**omega)**fp_len
        if strength == 1.0:
            false_miss = 1.0

        return false_miss