from attacks._base import Attack
import time
import random


class BitFlippingAttack(Attack):

    def __init__(self):
        super().__init__()

    """
    Runs the attack; gets a copy of 'dataset' with 'fraction' altered items
    fraction [0,1]
    """
    def run(self, dataset, fraction):
        start = time.time()
        # never alters the ID or the target (if provided)
        altered = dataset.copy()
        for i in range(int(fraction*(dataset.size-len(dataset)))):
            row = random.choice(dataset['Id'])
            column = random.choice(dataset.columns.drop(labels=["Id"]))
            value = dataset[column][row]
            if dataset[column].dtype == 'O':
                # categorical
                domain = list(set(dataset[column][:]))
                domain.remove(value)
                new_value = random.choice(domain)
            else:
                # numerical
                new_value = value ^ 1  # flipping the least significant bit
            altered.at[row, column] = new_value

        print("Bit-flipping attack runtime on " + str(fraction*100) + "% of entries: " +
              str(time.time() - start) + " sec.")
        return altered
