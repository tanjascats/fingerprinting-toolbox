from attacks._base import Attack
import time
import random


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
