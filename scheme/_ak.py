"""
AK Scheme
"""

from utils import *
import sys
import random
import time

from ._base import Scheme
# from ._base import LinearClassifierMixin, SparseCoefMixin, BaseEstimator
# from ..svm._base import _fit_liblinear

# placeholder for helper functions whose names start with underscore _


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

    def insertion(self, dataset, recipient_id, save=False, exclude=None, include=None, primary_key_attribute=None,
                  write_to=None):
        print(self._INIT_MESSAGE)
        # it is assumed that the first column in the dataset is the primary key
        # todo: a method that deals with importing the dataset
        # dataset may be passed as a dataset object or a string specfying the path or a pandas dataframe
        if type(dataset) != 'string':
            if primary_key_attribute is not None:
                primary_key = dataset[primary_key_attribute]
                original = dataset
                relation = dataset.drop(primary_key_attribute, axis=1)
                # relation.index = primary_key
            else:
                primary_key = dataset.index
                original = dataset
                relation = dataset
        else:
            relation, primary_key = import_dataset(dataset)
            original = relation
        # handle the attributes for marking

        if exclude is not None:
            for attribute in exclude:
                relation = relation.drop(attribute, axis=1)
        if include is not None:
            relation = relation[include]
        # number of numerical attributes
        num_of_attributes = len(relation.select_dtypes(exclude='object').columns)

        fingerprint = super().create_fingerprint(recipient_id)

        fingerprinted_relation = relation.copy()
        # count marked tuples
        count = count_omega = 0
        start = time.time()
        for r in relation.select_dtypes(exclude='object').iterrows():
            # seed = concat(secret_key, primary_key)
            seed = (self.secret_key << self.__primary_key_len) + primary_key[r[0]]
            random.seed(seed)

            # select the tuple
            if random.randint(0, sys.maxsize) % self.gamma == 0:
                # select attribute (that is not the primary key)
                attr_idx = random.randint(0, sys.maxsize) % num_of_attributes
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
                fingerprinted_relation.at[r[0], r[1].keys()[attr_idx]] = marked_attribute
                count += 1

        # put back the excluded stuff
        for attribute in exclude:
            fingerprinted_relation[attribute] = original[attribute]
        if primary_key_attribute is not None:
            fingerprinted_relation[primary_key_attribute] = original[primary_key_attribute]
        fingerprinted_relation = fingerprinted_relation[original.columns]
        print("Fingerprint inserted.")
        print("\tmarked tuples: ~" + str((count / len(relation)) * 100) + "%")
        print("\tsingle fingerprint bit embedded " + str(count_omega) + " times")
        if save and write_to is None:
            write_dataset(fingerprinted_relation, "ak_scheme", dataset, [self.gamma, self.xi], buyer_id)
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

    def detection(self, dataset, exclude=None, include=None, read=False, primary_key_attribute=None,
                  real_buyer_id=None):
        print("Start AK detection algorithm...")
        print("\tgamma: " + str(self.gamma) + "\n\txi: " + str(self.xi))
        if isinstance(dataset, pd.DataFrame) and read is False:
            relation = dataset
            # drop stuff
            if primary_key_attribute is not None:
                primary_key = relation[primary_key_attribute]
                relation = relation.drop(primary_key_attribute, axis=1)
            else:
                primary_key = relation.index
        elif read:  # then we assume that the dataset parameter is string name of the dataset or path or something
            relation, primary_key = import_fingerprinted_dataset(scheme_string="ak_scheme", dataset_name=dataset,
                                                                 scheme_params=[self.gamma, self.xi],
                                                                 real_buyer_id=real_buyer_id)
        if exclude is not None:
            relation = relation.drop(exclude, axis=1)

        start = time.time()
        # number of numerical attributes minus primary key
        num_of_attributes = len(relation.select_dtypes(exclude='object').columns)

        # init fingerprint template and counts
        # for each of the fingerprint bit the votes if it is 0 or 1
        count = [[0, 0] for x in range(self.fingerprint_bit_length)]

        # scan all tuples and obtain counts for each fingerprint bit
        for r in relation.select_dtypes(exclude='object').iterrows():
            seed = (self.secret_key << self.__primary_key_len) + primary_key[r[0]]
            random.seed(seed)

            # this tuple was marked
            if random.randint(0, sys.maxsize) % self.gamma == 0:
                # this attribute was marked (skip the primary key)
                attr_idx = random.randint(0, sys.maxsize) % num_of_attributes
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

        buyer_no = super().detect_potential_traitor(fingerprint_template_str)
        if buyer_no >= 0:
            print("Buyer " + str(buyer_no) + " is suspected.")
        else:
            print("None suspected.")
        print("Runtime: " + str(int(time.time() - start)) + " sec.")
        return buyer_no
