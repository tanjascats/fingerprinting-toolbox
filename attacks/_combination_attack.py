import pandas as pd

from attacks._base import Attack
from attacks import *
import time
import random
from sdv.lite import SingleTablePreset
import datasets

"""
In this attack, we:
    (i) perform SupersetWithDeletion attack, i.e. add synthetic record and remove the same number of original records, 
        such that the size of the resulting dataset stays the same as the original, and 
    (ii) perform FlippingAttack on top.
"""


class DeletionSupersetFlipping(Attack):

    def __init__(self):
        super().__init__()
    """
    Runs the attack; removes and creates strength_superset*data_size synthetic records, flips strngth_flipping*100% data 
    values. 
        strength[0,1]
    """
    def run(self, dataset, primary_key_attribute, strength_superset, strength_flipping, table_metadata,
            random_state=None):
        start = time.time()

        superset_attack = SupersetWithDeletion()
        stage1_dataset = superset_attack.run(dataset=dataset, primary_key_attribute=primary_key_attribute,
                                             strength=strength_superset, random_state=random_state,
                                             table_metadata=table_metadata)
        flipping_attack = FlippingAttack()
        result = flipping_attack.run(dataset=stage1_dataset,strength=strength_flipping,random_state=random_state)
        print("Combination attack: " + str(strength_superset) + " (superset with deletion) + " + str(strength_flipping)
              + " (flipping)\n\ttime: " + str(round(time.time() - start, 2)) + " sec.")
        return result

