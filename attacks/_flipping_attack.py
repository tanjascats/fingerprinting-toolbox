from attacks._base import Attack
import time
import random
from scipy.stats import binom
from datasets import Dataset



class FlippingAttack(Attack):

    def __init__(self):
        super().__init__()

    """
    Runs the attack; gets a copy of 'dataset' with 'strength' altered items
    strength [0,1]
    """
    def run(self, dataset, strength, random_state=None, keep_columns=None):
        start = time.time()
        # never alters the ID or the target (if provided)
        altered = dataset.copy()
        column_choice = dataset.columns
        if 'Id' in dataset.columns:
            column_choice = column_choice.drop(labels=["Id"])
        if keep_columns is not None:
            column_choice = column_choice.drop(labels=keep_columns)
        attribute_domain = {column: list(set(dataset[column][:])) for column in column_choice
                            if dataset[column].dtype == 'O'}
        # print('no. iterations: ' + str(int(strength * (dataset.size - len(dataset)))))
        for i in range(int(strength * (dataset.size - len(dataset)))):
            # iteration_start = time.time()
            random_state += (random_state+1)*(i+1)
            random.seed(random_state)
            row = random.choice(dataset.index)
            column = random.choice(column_choice)
            value = dataset[column][row]
            if dataset[column].dtype == 'O':
                # categorical
                domain = attribute_domain[column].copy()
                domain.remove(value)
                new_value = random.choice(domain)
                # timestamp_categorical = time.time()
                # print('---categorical iteration: ' + str(timestamp_categorical-iteration_start))
            else:
                # numerical
                new_value = value ^ 1  # flipping the least significant bit
                # timestamp_numerical = time.time()
                # print('---numerical iteration: ' + str(timestamp_numerical-iteration_start))
            altered.at[row, column] = new_value

        print("Flipping attack runtime on " + str(strength * 100) + "% of entries: " +
              str(round(time.time() - start, 2)) + " sec.")

        return altered

    def false_miss_estimation(self, dataset, strength, scheme):
        # calculates the cumulative binomial probability - estimation of attack SUCCESS
        # fm = 1- (1 - B(0.5*omega; omega, p=strenght)); omega = len(dataset)/(gamma*fp_len)
        if isinstance(dataset, Dataset):
            data_len = dataset.number_of_rows
        else:
            data_len = len(dataset)
        fp_len = scheme.get_fplen()
        gamma = scheme.get_gamma()
        omega = round(data_len / (gamma * fp_len), 0)

        # probability that the attack is successful
        b = 1-(binom.cdf(int(0.5*omega), omega, strength*(1/scheme.get_xi())) -
               binom.pmf(int(0.5*omega), omega, strength*(1/scheme.get_xi())))
        fm = 1 - pow(1 - b, fp_len)
        return fm

