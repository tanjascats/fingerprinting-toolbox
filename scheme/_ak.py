"""
AK Scheme
"""

from utils import *
from utils import _read_data
import sys
import random
import time

from ._base import Scheme


def _data_preprocess(dataset, exclude=None, include=None):
    '''
    Preprocess the data for fingerprinting with AK scheme. That includes:
    1) Remove user defined columns
    2) Remove the primary key
    3) Remove the target
    4) Remove non-numeric data for AK scheme
    The method changes the passed dataset
    :param dataset: Dataset instance
    :return: Dataset instance
    '''
    relation = dataset
    if exclude is not None:
        if not isinstance(exclude, list):
            print('Error! "exclude" parameter should be a list of attribute names')
            exit(1)
        for attribute in exclude:
            relation.set_dataframe(relation.dataframe.drop(attribute, axis=1))
        include = None
    if include is not None:
        relation.set_dataframe(relation.dataframe[include])
    relation.remove_primary_key()
    relation.remove_target()
    relation.remove_categorical()
    return relation


def _data_postprocess(fingerprinted_dataset, original_dataset):
    diff = original_dataset.columns.difference(fingerprinted_dataset.columns)
    for attribute in diff:
        fingerprinted_dataset.add_column(attribute, original_dataset.dataframe[attribute])
    fingerprinted_dataset.set_dataframe(fingerprinted_dataset.dataframe[original_dataset.dataframe.columns])
    return fingerprinted_dataset


class AKScheme(Scheme):
    """
    AK Scheme

    Parameters
    ----------

    """
    __primary_key_len = 20

    def __init__(self, gamma, fingerprint_bit_length=None, number_of_recipients=None, xi=1):
        self.gamma = gamma
        self.xi = xi

        if fingerprint_bit_length is not None:
            if number_of_recipients is not None:
                super().__init__(fingerprint_bit_length, number_of_recipients)
            else:
                super().__init__(fingerprint_bit_length)
        else:
            if number_of_recipients is not None:
                super().__init__(number_of_recipients)
            else:
                super().__init__()

        self._INIT_MESSAGE = "Start AK insertion algorithm...\n" \
                             "\tgamma: " + str(self.gamma) + "\n\txi: " + str(self.xi)

    def insertion(self, dataset, recipient_id, secret_key, save=False, exclude=None, include=None,
                  primary_key_attribute=None, target_attribute=None, write_to=None):
        print(self._INIT_MESSAGE)

        original_data = _read_data(dataset, target_attribute=target_attribute,
                                   primary_key_attribute=primary_key_attribute)
        # prep data for fingerprinting

        # relation is original data but preprocessed
        relation = original_data.clone()
        relation = _data_preprocess(dataset=relation, exclude=exclude, include=include)
        # fingerprinted_relation is a deep copy of an original and will be modified throughout the insertion phase
        fingerprinted_relation = original_data.clone()
        fingerprinted_relation = _data_preprocess(dataset=fingerprinted_relation, exclude=exclude, include=include)
        fingerprint = super().create_fingerprint(recipient_id, secret_key)

        # count marked tuples
        count = count_omega = 0
        start = time.time()
        for r in relation.dataframe.iterrows():
            # seed = concat(secret_key, primary_key)
            seed = (secret_key << self.__primary_key_len) + relation.primary_key[r[0]]
            random.seed(seed)

            # select the tuple
            if random.randint(0, sys.maxsize) % self.gamma == 0:
                # select attribute (that is not the primary key)
                attr_idx = random.randint(0, sys.maxsize) % relation.number_of_columns
                attribute_val = r[1][attr_idx]
                # select least significant bit
                bit_idx = random.randint(0, sys.maxsize) % self.xi
                # select mask bit
                mask_bit = random.randint(0, sys.maxsize) % 2
                # select fingerprint bit
                fingerprint_idx = random.randint(0, sys.maxsize) % self.fingerprint_bit_length
                if fingerprint_idx == 0:
                    count_omega += 1
                fingerprint_bit = fingerprint[fingerprint_idx]
                mark_bit = (mask_bit + fingerprint_bit) % 2

                # alter the chosen value
                marked_attribute = set_bit(attribute_val, bit_idx, mark_bit)
                fingerprinted_relation.dataframe.at[r[0], r[1].keys()[attr_idx]] = marked_attribute
                count += 1

        # put back the excluded stuff
        fingerprinted_relation = _data_postprocess(fingerprinted_relation, original_data)
        print("Fingerprint inserted.")
        print("\tmarked tuples: ~" + str(round((count / relation.number_of_rows), 4) * 100) + "%")
        print("\tsingle fingerprint bit embedded " + str(count_omega) + " times")
        if save and write_to is None:
            fingerprinted_relation.save("ak_scheme_{}_{}_{}".format(self.gamma, self.xi, recipient_id))
            #write_dataset(fingerprinted_relation, "ak_scheme", dataset, [self.gamma, self.xi], recipient_id)
            # todo: this will break
        elif save and write_to is not None:
            write_to_dir = "/".join(write_to.split("/")[:-1])
            if not os.path.exists(write_to_dir):
                os.mkdir(write_to_dir)
            fullname = os.path.join(write_to)
            fingerprinted_relation.to_csv(fullname)
        runtime = int(time.time() - start)
        if runtime == 0:
            runtime_string = "<1"
        else:
            runtime_string = str(runtime)
        print("Time: " + runtime_string + " sec.")
        return fingerprinted_relation
    # todo: return fingerprint meta, i.e. gamma, xi, excluded attributes, primary key attribute, number of recipients

    def detection(self, dataset, secret_key, exclude=None, include=None, read=False, primary_key_attribute=None,
                  real_recipient_id=None):
        print("Start AK detection algorithm...")
        print("\tgamma: " + str(self.gamma) + "\n\txi: " + str(self.xi))
        fingerprinted_data = _read_data(dataset)
        fingerprinted_data_prep = fingerprinted_data.clone()
        fingerprinted_data_prep = _data_preprocess(fingerprinted_data_prep, exclude=exclude, include=include)

        start = time.time()
        # init fingerprint template and counts
        # for each of the fingerprint bit the votes if it is 0 or 1
        count = [[0, 0] for x in range(self.fingerprint_bit_length)]

        # scan all tuples and obtain counts for each fingerprint bit
        for r in fingerprinted_data_prep.dataframe.iterrows():
            seed = (secret_key << self.__primary_key_len) + fingerprinted_data_prep.primary_key[r[0]]
            random.seed(seed)

            # this tuple was marked
            if random.randint(0, sys.maxsize) % self.gamma == 0:
                # this attribute was marked (skip the primary key)
                attr_idx = random.randint(0, sys.maxsize) % fingerprinted_data_prep.number_of_columns
                attribute_val = r[1][attr_idx]
                # this LS bit was marked
                bit_idx = random.randint(0, sys.maxsize) % self.xi
                # take care of negative values
                if attribute_val < 0:
                    attribute_val = -attribute_val
                    # raise flag
                mark_bit = (attribute_val >> bit_idx) % 2
                mask_bit = random.randint(0, sys.maxsize) % 2
                # fingerprint bit = mark_bit xor mask_bit
                fingerprint_bit = (mark_bit + mask_bit) % 2
                fingerprint_idx = random.randint(0, sys.maxsize) % self.fingerprint_bit_length
                # update votes
                count[fingerprint_idx][fingerprint_bit] += 1

        # this fingerprint template will be upside-down from the real binary representation
        fingerprint_template = [2] * self.fingerprint_bit_length
        # recover fingerprint
        for i in range(self.fingerprint_bit_length):
            # certainty of a fingerprint value
            T = 0.50
            try:
                if count[i][0]/(count[i][0] + count[i][1]) > T:
                    fingerprint_template[i] = 0
                elif count[i][1]/(count[i][0] + count[i][1]) > T:
                    fingerprint_template[i] = 1
            except ZeroDivisionError:
                pass

        fingerprint_template_str = ''.join(map(str, fingerprint_template))
        print("Fingerprint detected: " + list_to_string(fingerprint_template))

        recipient_no = super().detect_potential_traitor(fingerprint_template_str, secret_key)
        if recipient_no >= 0:
            print("Buyer " + str(recipient_no) + " is suspected.")
        else:
            print("None suspected.")
        print("Runtime: " + str(int(time.time() - start)) + " sec.")
        return recipient_no
