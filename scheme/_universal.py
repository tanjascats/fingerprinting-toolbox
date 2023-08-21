import json
import sys
import random

from utils import *
from utils import _read_data

from ._base import Scheme
from pprint import pprint

def _data_preprocess(dataset, exclude=None, include=None):
    '''
    Preprocess the data for fingerprinting with Universal scheme. That includes:
    1) Remove user defined columns
    2) Remove the primary key
    3) Remove the target
    4) Number-encode the categorical attributes
    The method changes the passed dataset
    :param dataset: Dataset instance
    :return: Dataset instance
    '''
    relation = dataset
    if exclude is not None:
        for attribute in exclude:
            try:
                relation.set_dataframe(relation.dataframe.drop(attribute, axis=1))
            except KeyError:
                print('Warning! Trying to exclude non existing columns:{}'.format(attribute))
            relation.set_dataframe(relation.dataframe.drop(attribute, axis=1))
        include = None
    if include is not None:
        relation.set_dataframe(relation.dataframe[include])
    relation.remove_primary_key()
    relation.remove_target()
    relation.columns = relation.dataframe.columns
    relation.number_of_rows, relation.number_of_columns = relation.dataframe.shape
    relation._set_types()

    relation.number_encode_categorical()
    return relation


def _data_postprocess(fingerprinted_dataset, original_dataset):
    '''
    Processes data after fingerprinting. This includes:
    1) Retrieving all the original columns in the correct order
    2) Decode the categorical values
    :param fingerprinted_dataset: datasets.Dataset instance
    :param original_dataset: datasets.Dataset instance
    :return: datasets.Dataset instance of processed fingerprinted dataset ready to be published
    '''
    diff = original_dataset.columns.difference(fingerprinted_dataset.columns)
    for attribute in diff:
        fingerprinted_dataset.add_column(attribute, original_dataset.dataframe[attribute])
    fingerprinted_dataset.set_dataframe(fingerprinted_dataset.dataframe[original_dataset.dataframe.columns])
    # learn the label encoder on original and apply on fingerprinted numerical
    fingerprinted_dataset.decode_categorical()
    # fix decimal types
    for dec in original_dataset.decimal_attributes:
        fingerprinted_dataset.dataframe[dec] = pd.to_numeric(fingerprinted_dataset.dataframe[dec])
    return fingerprinted_dataset


class Universal(Scheme):
    '''
    Fingerprinting scheme applicable to all data types. Unifies the AK scheme for numerical data and adapted AK for
    categorical and decimal data.
        - categorical data: sorted, encoded to numerical and modified with constraints
        - decimal data: least significant decimal place is modified
    '''
    __primary_key_len = 20

    def __init__(self, gamma, fingerprint_bit_length=None, number_of_recipients=None, xi=1):
        self.gamma = gamma
        self.xi = xi

        self.detection_counts = None

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

        self._INIT_MESSAGE = "Universal fingerprinting scheme - initialised.\nEmbedding started...\n" \
                             "\tgamma: " + str(self.gamma) + "\n\tfingerprint length: " + \
                             str(self.fingerprint_bit_length) + "\n\txi: " + str(xi) + \
                             "\n\t# recipients: " + str(number_of_recipients)
        self.original_attributes = None

    def insertion(self, dataset, recipient_id, secret_key, save=False, exclude=None, include=None,
                  primary_key_attribute=None, target_attribute=None, write_to=None, attributes_weights=None):
        '''
        Embeds the fingerprint into the data.
        :param dataset: data path, Pandas dataframe or datasets.Dataset class instance
        :param recipient_id: Recipient ID
        :param secret_key: Owner's secret key
        :param save: If True, save the fingerprinted data to file <scheme_name>_<gamma>_<fingerprint_bit_length>_<recipient_id>.csv
        :param exclude: List of columns to exclude from fingerprinting
        :param include: List of columns to include to fingerprinting (this is ignored if excluded is defined)
        :param primary_key_attribute: Name of the primary key attribute; optional
        :param target_attribute: Name of the target attribute; optional
        :param write_to: Name of the target datafile; if defined, 'save' is ignored
        :param attributes_weights: List of attributes weights that should sum up to 1, alter less relevant attributes more; optional
        :return: datasets.Dataset instance of fingerprinted data
        '''
        print(self._INIT_MESSAGE)
        print("\n\t(secret key -- for evaluation purposes): " + str(secret_key))
        original_data = _read_data(dataset, target_attribute=target_attribute,
                                  primary_key_attribute=primary_key_attribute)
        # prep data for fingerprinting

        # relation is original data but preprocessed
        relation = original_data.clone()
        relation = _data_preprocess(dataset=relation, exclude=exclude, include=include)
        # fingerprinted_relation is a deep copy of an original and will be modified throughout the insertion phase
        self.original_attributes = relation.columns

        fingerprinted_relation = original_data.clone()
        fingerprinted_relation = _data_preprocess(dataset=fingerprinted_relation, exclude=exclude, include=include)
        fingerprint = super().create_fingerprint(recipient_id, secret_key)

        # count marked tuples
        count = count_omega = 0
        count_omega = [0 for i in range(self.fingerprint_bit_length)]
        start = time.time()
        print('\tInserting a fingerprint into columns: ' + str(relation.dataframe.columns))

        for r in relation.dataframe.iterrows():
            # seed = concat(secret_key, primary_key)
            seed = int((secret_key << self.__primary_key_len) + relation.primary_key[r[0]])
            random.seed(seed)

            # select the tuple
            if random.choices([0, 1], [1/self.gamma, 1-1/self.gamma]) == [0]:
                # select attribute (that is not the primary key)
                if attributes_weights is not None:
                    attr_idx = random.choices(range(relation.number_of_columns), weights=[1-x for x in attributes_weights])[0]
                else:
                    attr_idx = random.randint(0, sys.maxsize) % relation.number_of_columns
                attribute_val = r[1][attr_idx]
                # select least significant bit
                bit_idx = random.randint(0, sys.maxsize) % self.xi
                # select mask bit
                mask_bit = random.randint(0, sys.maxsize) % 2
                # select fingerprint bit
                fingerprint_idx = random.randint(0, sys.maxsize) % self.fingerprint_bit_length
                count_omega[fingerprint_idx] += 1
                fingerprint_bit = fingerprint[fingerprint_idx]
                mark_bit = (mask_bit + fingerprint_bit) % 2
                # if the value is categorical, the mark_bit indicates whether the new value will be odd or even
                decimal_places = None
                if fingerprinted_relation.dataframe.columns[attr_idx] in fingerprinted_relation.categorical_attributes:
                    all_vals = fingerprinted_relation.get_distinct(attr_idx)
                    if mark_bit == 1:  # odd value
                        possible_vals = [v for v in all_vals if v % 2 == 1]
                    else:
                        possible_vals = [v for v in all_vals if v % 2 == 0]
                    marked_attribute = random.choice(possible_vals)
                elif fingerprinted_relation.dataframe.columns[attr_idx] in fingerprinted_relation.decimal_attributes:
                    attribute_val_str = str(attribute_val)
                    decimal_places = attribute_val_str[::-1].find('.')
                    attribute_val = round(attribute_val * (10**decimal_places))
                    # alter the chosen value
                    marked_attribute = set_bit(attribute_val, bit_idx, mark_bit)
                    # return the decimal
                    if decimal_places > 0:
                        marked_attribute = marked_attribute / (10**decimal_places)
                else:
                    decimal_places = 0
                    attribute_val = round(attribute_val)
                    marked_attribute = set_bit(attribute_val, bit_idx, mark_bit)
                if decimal_places is not None:
                    if decimal_places == 2:
                        fingerprinted_relation.dataframe.at[r[0], r[1].keys()[attr_idx]] = f'{marked_attribute:.2f}'
                    elif decimal_places == 3:
                        fingerprinted_relation.dataframe.at[r[0], r[1].keys()[attr_idx]] = f'{marked_attribute:.3f}'
                    elif decimal_places == 4:
                        fingerprinted_relation.dataframe.at[r[0], r[1].keys()[attr_idx]] = f'{marked_attribute:.4f}'
                    else:
                        fingerprinted_relation.dataframe.at[r[0], r[1].keys()[attr_idx]] = marked_attribute
                else:
                    fingerprinted_relation.dataframe.at[r[0], r[1].keys()[attr_idx]] = round(marked_attribute)
                count += 1

        # put back the excluded stuff
        fingerprinted_relation = _data_postprocess(fingerprinted_relation, original_data)
        print("Fingerprint inserted.")
        print("\tmarked tuples: ~" + str(round((count / relation.number_of_rows) * 100, 2)) + "%")
        print("\tsingle fingerprint bit embedded " + str(int(np.mean(count_omega))) + " times (\"amount of "
                                                                                      "redundancy\")")
        if save and write_to is None:
            fingerprinted_relation.save("universal_scheme_{}_{}_{}.csv".format(self.gamma, self.fingerprint_bit_length,
                                                                        recipient_id))
        elif write_to is not None:
            write_to_dir = "/".join(write_to.split("/")[:-1])
            if not os.path.exists(write_to_dir):
                os.mkdir(write_to_dir)
            fullname = os.path.join(write_to)
            fingerprinted_relation.dataframe.to_csv(fullname, index=False)
        runtime = int(time.time() - start)
        if runtime == 0:
            runtime_string = "<1"
        else:
            runtime_string = str(runtime)
        print("Time: " + runtime_string + " sec.")
        return fingerprinted_relation

    def detection(self, dataset, secret_key, exclude=None, include=None, primary_key_attribute=None,
                  target_attribute=None, attributes_weights=None, original_attributes=None, save_counts_path=None):
        '''
        Detects the fingerprint from the data and assigns a suspect.
        :param dataset: path, pandas.DataFrame or Dataset instance of the suspicious dataset
        :param secret_key: owner's secret key
        :param exclude: list of column names excluded from fingerprinting
        :param include: list of column names included in fingerprinting (ignored if 'exclude' is provided)
        :param primary_key_attribute: optional, name of the primary key attribute
        :param target_attribute: optional; name of the target attribute
        :param attributes_weights: List of attributes weights that should sum up to 1, alter less relevant attributes more; optional
        :param save_counts_path: path to file where the bit counts will be written
        :return: suspected recipient ID
        '''
        print("Start detection algorithm...")
        print("\tgamma: " + str(self.gamma) + "\n\tfingerprint length: " + str(self.fingerprint_bit_length))
        fingerprinted_data = _read_data(dataset)
        fingerprinted_data_prep = fingerprinted_data.clone()
        if isinstance(dataset, Dataset):
            target_attribute = dataset.target_attribute
            primary_key_attribute = dataset.primary_key_attribute
        if target_attribute is not None:
            fingerprinted_data_prep._set_target_attribute = target_attribute
        if primary_key_attribute is not None:
            fingerprinted_data_prep._set_primary_key(primary_key_attribute)
        if original_attributes is not None:
            self.original_attributes = original_attributes
        fingerprinted_data_prep = _data_preprocess(fingerprinted_data_prep, exclude=exclude, include=include)
        # return the original attribute list and fill out the missing with zeroes in case of vertical attack
        # todo: this is terribly wrong but works for now
        # print(fingerprinted_data.dataframe)
        # print(fingerprinted_data.dataframe.drop([primary_key_attribute,target_attribute], axis=1).columns.to_list())
        # print(self.original_attributes.to_list())
        if self.original_attributes is not None:
            if primary_key_attribute is not None:
                # if not fingerprinted_data.dataframe.drop([primary_key_attribute,target_attribute], axis=1).columns.to_list() == self.original_attributes.to_list():
                if not len(fingerprinted_data.dataframe.drop([primary_key_attribute, target_attribute],
                                                         axis=1).columns.to_list()) == len(self.original_attributes):
                    print('Vertical attack detected. The detection might have a reduced success.')
                    # try:
                    if exclude is None:
                        _exclude = []
                    else:
                        _exclude = exclude
                    difference = set(self.original_attributes).difference(set(_exclude)).difference(
                        set(fingerprinted_data_prep.columns))
                    for diff in difference:
                        fingerprinted_data_prep.dataframe[diff] = pd.Series(data=[0 for i in
                                                                                  range(
                                                                                      len(fingerprinted_data_prep.dataframe))])
                    # original_order = self.original_attributes.to_list()
                    original_order = self.original_attributes
                    for el in _exclude:
                        original_order.remove(el)
                    fingerprinted_data_prep.set_dataframe(fingerprinted_data_prep.dataframe[original_order])
                    # except AttributeError:
                    #    print('\nWARNING!\n\t->Provide the original attribute names, if available, to improve the '
                    #          'performance of detection algorithm.\n')
                    # if not relation_orig.columns.equals(relation_fp.columns):
                    #    print(relation_fp.columns)
                    #    difference = relation_orig.columns.difference(relation_fp.columns)
                    #    for diff in difference:
                    #        relation_fp[diff] = relation_orig[diff]
                    # bring back the original order of columns
                    # relation_fp = relation_fp[relation_orig.columns.tolist()]
            else:
                if not fingerprinted_data.dataframe.drop([target_attribute],
                                                         axis=1).columns.to_list() == self.original_attributes.to_list():
                    print('Vertical attack detected. The detection might have a reduced success.')
                    # try:
                    if exclude is None:
                        _exclude = []
                    else:
                        _exclude = exclude
                    difference = set(self.original_attributes).difference(set(_exclude)).difference(
                        set(fingerprinted_data_prep.columns))
                    for diff in difference:
                        fingerprinted_data_prep.dataframe[diff] = pd.Series(data=[0 for i in
                                                                                  range(
                                                                                      len(fingerprinted_data_prep.dataframe))])
                    original_order = self.original_attributes.to_list()
                    for el in _exclude:
                        original_order.remove(el)
                    fingerprinted_data_prep.set_dataframe(fingerprinted_data_prep.dataframe[original_order])
        start = time.time()
        # init fingerprint template and counts
        # for each of the fingerprint bit the votes if it is 0 or 1
        count = [[0, 0] for x in range(self.fingerprint_bit_length)]

        # scan all tuples and obtain counts for each fingerprint bit
        for r in fingerprinted_data_prep.dataframe.iterrows():
            seed = int((secret_key << self.__primary_key_len) + fingerprinted_data_prep.primary_key[r[0]])
            random.seed(seed)

            # this tuple was marked
            if random.choices([0, 1], [1 / self.gamma, 1 - 1 / self.gamma]) == [0]:
            # if random.randint(0, sys.maxsize) % self.gamma == 0:
                # this attribute was marked (skip the primary key)
                if attributes_weights is not None:
                    attr_idx = random.choices(range(fingerprinted_data_prep.number_of_columns), weights=[1-x for x in attributes_weights])[0]
                else:
                    attr_idx = random.randint(0, sys.maxsize) % fingerprinted_data_prep.number_of_columns
                attribute_val = r[1][attr_idx]
                # this LS bit was marked
                bit_idx = random.randint(0, sys.maxsize) % self.xi
                # take care of negative values
                if attribute_val < 0:
                    attribute_val = -attribute_val
                    # raise flag
                if fingerprinted_data_prep.dataframe.columns[attr_idx] in fingerprinted_data_prep.categorical_attributes:
                    attribute_val = round(attribute_val)
                attribute_val_str = str(attribute_val)
                decimal_places = attribute_val_str[::-1].find('.')
                if fingerprinted_data_prep.dataframe.columns[attr_idx] in fingerprinted_data_prep.integer_attributes:
                    decimal_places = 0
                if decimal_places == -1:
                    decimal_places = 0
                attribute_val = round(attribute_val * (10 ** decimal_places))

                if fingerprinted_data_prep.dataframe.columns[attr_idx] in fingerprinted_data_prep.categorical_attributes:
                    if attribute_val % 2 == 0:
                        mark_bit = 0
                    else:
                        mark_bit = 1
                else:
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
        print("Potential fingerprint detected: " + list_to_string(fingerprint_template))
        self.detection_counts = count
        print('Detection counts:')
        print(self.detection_counts)#
        if save_counts_path is not None:
            with open(save_counts_path, 'w') as outfile:
                json.dump([self.fingerprint_bit_length, self.gamma, self.xi, secret_key], outfile)
                json.dump(self.detection_counts, outfile)

        recipient_no = super().detect_potential_traitor(fingerprint_template_str, secret_key)
        if recipient_no >= 0:
            print("Recipient " + str(recipient_no) + " is suspected.")
        else:
            print("No one suspected.")
        print("Runtime: " + str(int(time.time() - start)) + " sec.")
        return recipient_no

    def get_gamma(self):
        return self.gamma

    def get_xi(self):
        return self.xi

    def get_fplen(self):
        return self.fingerprint_bit_length

    def to_string(self):
        return 'universal'