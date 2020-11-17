from attacks.attack import Attack
import time
import random


class CombinedAttack(Attack):

    def __init__(self):
        super().__init__()

    """
    Runs the attack
    Returns a dataset of fraction_subset*size rows,
                         without columns <columns> and
                         fraction_flipping amount of flipped values of the remaining data set
    """
    def run(self, dataset, fraction_subset, number_of_columns, fraction_flipping, random_seed=None):
        if random_seed is not None:
            random.seed(random)
        # 1) horizontal attack
        if fraction_subset < 0 or fraction_subset > 1:
            return None

        start = time.time()
        subset_horizontally = dataset.sample(frac=fraction_subset)

        # 2) vertical attack
        if number_of_columns >= len(subset_horizontally.columns)-1:
            print("Cannot delete all columns.")
            return None

        column_subset = random.choices(subset_horizontally.columns.drop(labels=["Id"]), k=number_of_columns)
        subset_vertically = subset_horizontally.drop(column_subset, axis=1)

        # 3) bit flipping attack
        altered = subset_vertically.copy()
        for i in range(int(fraction_flipping * (subset_vertically.size - len(subset_vertically)))):
            row = random.choice(subset_vertically['Id'].values)
            column = random.choice(subset_vertically.columns.drop(labels=["Id"]))
            value = subset_vertically[column][row]
            if subset_vertically[column].dtype == 'O':
                # categorical
                domain = list(set(subset_vertically[column][:]))
                domain.remove(value)
                new_value = random.choice(domain)
            else:
                # numerical
                new_value = value ^ 1  # flipping the least significant bit
            altered.at[row, column] = new_value

        return altered