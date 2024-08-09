"""
AHK Scheme

R. Agrawal, P.J. Haas, and J. Kiernan, “Watermarking Relational
Data: Framework, Algorithms and Analysis,” The VLDB J., vol. 12,
no. 2, pp. 157-169, 2003
"""

from utils import *
from utils import _read_data
import sys
import random
import time

from ._base import Scheme


def _data_preprocess(dataset, exclude=None, include=None):
    '''
    Preprocess the data for watermarking with AK scheme. That includes:
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


def _data_postprocess(watermarked_dataset, original_dataset):
    diff = original_dataset.columns.difference(watermarked_dataset.columns)
    for attribute in diff:
        watermarked_dataset.add_column(attribute, original_dataset.dataframe[attribute])
    watermarked_dataset.set_dataframe(watermarked_dataset.dataframe[original_dataset.dataframe.columns])
    return watermarked_dataset


class AKScheme(Scheme):
    """
    AK Scheme

    Parameters
    ----------

    """
    __primary_key_len = 20

    def __init__(self, gamma, watermark_bit_length=None, number_of_recipients=None, xi=1):
        self.gamma = gamma
        self.xi = xi

        if watermark_bit_length is not None:
            if number_of_recipients is not None:
                super().__init__(watermark_bit_length, number_of_recipients)
            else:
                super().__init__(watermark_bit_length)
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
        # prep data for watermarking

        # relation is original data but preprocessed
        relation = original_data.clone()
        relation = _data_preprocess(dataset=relation, exclude=exclude, include=include)
        # watermarked_relation is a deep copy of an original and will be modified throughout the insertion phase
        watermarked_relation = original_data.clone()
        watermarked_relation = _data_preprocess(dataset=watermarked_relation, exclude=exclude, include=include)

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
                # select watermark bit
                watermark_bit = random.randint(0, sys.maxsize) % 2

                # alter the chosen value
                marked_attribute = set_bit(attribute_val, bit_idx, watermark_bit)
                watermarked_relation.dataframe.at[r[0], r[1].keys()[attr_idx]] = marked_attribute
                count += 1

        # put back the excluded stuff
        watermarked_relation = _data_postprocess(watermarked_relation, original_data)
        print("watermark inserted.")
        print("\tmarked tuples: ~" + str(round((count / relation.number_of_rows), 4) * 100) + "%")
        print("\tsingle watermark bit embedded " + str(count_omega) + " times")
        if save and write_to is None:
            watermarked_relation.save("ak_scheme_{}_{}_{}".format(self.gamma, self.xi, recipient_id))
            #write_dataset(watermarked_relation, "ak_scheme", dataset, [self.gamma, self.xi], recipient_id)
            # todo: this will break
        elif save and write_to is not None:
            write_to_dir = "/".join(write_to.split("/")[:-1])
            if not os.path.exists(write_to_dir):
                os.mkdir(write_to_dir)
            fullname = os.path.join(write_to)
            watermarked_relation.to_csv(fullname)
        runtime = int(time.time() - start)
        if runtime == 0:
            runtime_string = "<1"
        else:
            runtime_string = str(runtime)
        print("Time: " + runtime_string + " sec.")
        return watermarked_relation
    # todo: return watermark meta, i.e. gamma, xi, excluded attributes, primary key attribute, number of recipients

    def detection(self, dataset, secret_key, exclude=None, include=None, read=False, primary_key_attribute=None,
                  target_attribute=None, real_recipient_id=None):
        print("Start AK detection algorithm...")
        print("\tgamma: " + str(self.gamma) + "\n\txi: " + str(self.xi))
        watermarked_data = _read_data(dataset)
        watermarked_data_prep = watermarked_data.clone()
        if target_attribute is not None:
            watermarked_data_prep._set_target_attribute = target_attribute
        if primary_key_attribute is not None:
            watermarked_data_prep._set_primary_key(primary_key_attribute)
        watermarked_data_prep = _data_preprocess(watermarked_data_prep, exclude=exclude, include=include)

        start = time.time()
        # init watermark template and counts
        # for each of the watermark bit the votes if it is 0 or 1

        total_count = 0
        match_count = 0
        # scan all tuples and obtain counts for each watermark bit
        for r in watermarked_data_prep.dataframe.iterrows():
            seed = (secret_key << self.__primary_key_len) + watermarked_data_prep.primary_key[r[0]]
            random.seed(seed)

            # this tuple was marked
            if random.randint(0, sys.maxsize) % self.gamma == 0:
                # this attribute was marked (skip the primary key)
                attr_idx = random.randint(0, sys.maxsize) % watermarked_data_prep.number_of_columns
                attribute_val = r[1][attr_idx]
                # this LS bit was marked
                bit_idx = random.randint(0, sys.maxsize) % self.xi
                # take care of negative values
                if attribute_val < 0:
                    attribute_val = -attribute_val
                    # raise flag
                check_bit = (attribute_val >> bit_idx) % 2
                watermark_bit = random.randint(0, sys.maxsize) % 2
                # update votes
                total_count += 1
                if check_bit == watermark_bit:
                    match_count += 1

        # todo: threshold is calculated differently
        T = 0.1 * total_count
        if match_count < T or match_count > total_count - T:
            print('piracy suspected')
            detected = 1  # watermark detected
        else:
            detected = -1  # watermark not detected
        return detected

# todo: extensions (see MarkBench/method_overview.md)