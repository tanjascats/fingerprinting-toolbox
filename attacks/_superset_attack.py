import pandas as pd

from attacks._base import Attack
import time
import random
from sdv.lite import SingleTablePreset
import datasets

"""
In this attack, we add synthetic record and remove the same number of original record, such that the size of the 
resulting dataset stays the same as the original.
"""


class SupersetWithDeletion(Attack):

    def __init__(self):
        super().__init__()
    """
    Runs the attack; removes and creates strength*data_size sythetic records
        strength[0,1]
    """
    def run(self, dataset, primary_key_attribute, strength, table_metadata, random_state=None):
        if strength < 0 or strength > 1:
            return None

        start = time.time()
        if random_state is None:
            random_state = int(start)
        else:
            random_state = random_state
        # assign auxiliary index for sorting
        dataset['aux_index'] = [i for i in range(len(dataset))]

        # - perform subset attack
        subset = dataset.sample(frac=1.0-strength, random_state=random_state)

        # - perform synthetisation
        metadata = table_metadata
        synthesizer = SingleTablePreset(metadata, name='FAST_ML')
        synthesizer.fit(data=dataset.drop('aux_index', axis=1))
        synthetic_data = synthesizer.sample(num_rows=len(dataset)-len(subset))

        # - insert synthetic record on deleted primary keys positions
        diff = set(dataset['aux_index']) - set(subset['aux_index'])
        synthetic_data['aux_index'] = list(diff)
        result = subset.append(synthetic_data, ignore_index=True)
        result.sort_values(by='aux_index', inplace=True)
        result.reset_index(inplace=True)
        result = result.drop(['index', 'aux_index'], axis=1)
        result[primary_key_attribute] = dataset[primary_key_attribute]
        print("Superset attack (with deletion) runtime on " + str(int(strength*len(dataset))) + " out of " +
              str(len(dataset)) + " entries: " + str(time.time()-start) + " sec.")
        return result


