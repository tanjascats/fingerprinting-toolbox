import time
import sys
import random
import operator
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import BallTree

from utils import *
from utils import _read_data
from ._base import Scheme


def _count_attributes(data, first_col_index=True):
    tot_attributes = len(data.columns)
    if first_col_index:
        tot_attributes -= 1
    return tot_attributes


def _train_balltrees(data, attr):
    """
    Trains balltrees for data and given attributes
    :param data: dataframe
    :param attr: a list of correlated attributes
    :return: dictionary of balltrees
    """
    start_training_balltrees = time.time()
    balltree = dict()
    for i in range(len(attr)):
        balltree_i = BallTree(data[attr[:i].append(attr[(i + 1):])], metric="hamming")
        balltree[attr[i]] = balltree_i
    balltree_all = BallTree(data[attr], metric="hamming")
    balltree["all"] = balltree_all
    print("Training balltrees in: " + str(round(time.time() - start_training_balltrees, 2)) + " sec.")
    return balltree


def _data_row_seed(row_index, secret_key):
    """
    Creates the seed for the specific data row and seeds the pseudo-random generator.
    :return:
    """
    # supports the dataset size of up to 1,048,576 entries
    _primary_key_len = 20
    seed = (secret_key << _primary_key_len) + row_index
    random.seed(seed)
    return random


def _select_marking_attribute(data_row, attribute_count):
    """
    Selects the attribute from the row fit for marking.
    :param data_row:
    :param attribute_count:
    :return: attribute name, attribute value
    """
    attr_idx = random.randint(0, sys.maxsize) % attribute_count + 1
    attr_name = data_row.index[attr_idx]
    attribute_val = data_row[attr_idx]
    return attr_idx, attr_name, attribute_val


def _select_mark(fingerprint):
    """
    Selects the mark according to the pseudo-random fingerprint bit.
    :param fingerprint:
    :return: mark bit
    """
    fingerprint_idx = random.randint(0, sys.maxsize) % len(fingerprint)
    fingerprint_bit = fingerprint[fingerprint_idx]
    # select mask and calculate the mark bit
    _mask_bit = random.randint(0, sys.maxsize) % 2
    _mark_bit = (_mask_bit + fingerprint_bit) % 2
    return _mark_bit


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
            relation.set_dataframe(relation.dataframe.drop(attribute, axis=1))
        include = None
    if include is not None:
        relation.set_dataframe(relation.dataframe[include])
    relation.remove_primary_key()
    relation.remove_target()
    relation.number_encode_categorical()
    return relation


def _data_postprocess(fingerprinted_dataset, original_dataset):
    diff = original_dataset.columns.difference(fingerprinted_dataset.columns)
    for attribute in diff:
        fingerprinted_dataset.add_column(attribute, original_dataset.dataframe[attribute])
    fingerprinted_dataset.set_dataframe(fingerprinted_dataset.dataframe[original_dataset.dataframe.columns])
    # todo: decode to categorical
    # learn the label encoder on original and apply on fingerprinted numerical
    fingerprinted_dataset.decode_categorical()
    return fingerprinted_dataset


class BNNScheme(Scheme):
    """
    Blind scheme for fingerprinting integer and categorical values based on nearest neighbourhood search.
    """

    def __init__(self, gamma, xi, fingerprint_bit_length=32, secret_key=333, number_of_recipients=10,
                 distance_based=False,
                 d=0, k=10):
        self.gamma = gamma
        self.xi = xi
        self.distance_based = distance_based  # if False, then fixed-size-neighbourhood-based with k=10 - default
        self.correlated_attributes = None
        if distance_based:
            self.d = d
        else:
            self.k = k
        super().__init__(fingerprint_bit_length, secret_key, number_of_recipients)

    def insertion(self, dataset, recipient_id, secret_key=None, save=False, force_change=False, marking_randomness=False):
        init_msg = "Start the insertion algorithm of a scheme for fingerprinting categorical data (neighbourhood) ..." \
                   "\n\tgamma: " + str(self.gamma) + "\n\txi: " + str(self.xi)
        print(init_msg)
        if secret_key is not None:
            self.secret_key = secret_key
        # it is always assumed that the first column in the dataset is the primary key !
        relation, primary_key = import_dataset(dataset)
        tot_attributes = _count_attributes(relation)

        fingerprint = super().create_fingerprint(recipient_id)

        start = time.time()

        # label encoder for categorical attributes
        # todo: maybe turn into a function
        categorical_attributes = relation.select_dtypes(include='object').columns
        label_encoders = dict()
        for cat in categorical_attributes:
            label_enc = LabelEncoder()
            relation[cat] = label_enc.fit_transform(relation[cat])
            label_encoders[cat] = label_enc

        # ball trees from user-specified correlated attributes
        self.correlated_attributes = categorical_attributes[
                                     :]  # todo: this is a demo set where we consider everything mutually correlated

        balltree = _train_balltrees(relation, self.correlated_attributes)

        fingerprinted_relation = relation.copy()
        # seed the row process
        # select the row if it satisfies pseudorandom criterium
        #
        for r in relation.iterrows():
            # r[0] is an index of a row = primary key
            # seed = concat(secret_key, primary_key)
            # todo: possible fix if it doesn't work properly: random = _data_row_seed(r[0], self.secret_key)
            _data_row_seed(r[0], self.secret_key)

            # selecting the tuple
            # todo: if row_selected: function
            _row_selected = (random.randint(0, sys.maxsize) % self.gamma == 0)
            if _row_selected:
                # selecting the attribute
                _attr_idx, _attr_name, _attr_val = _select_marking_attribute(r[1], tot_attributes)

                # select fingerprint bit
                _mark_bit = _select_mark(fingerprint)

                # check if attribute is categorical
                if _attr_name in categorical_attributes:
                    # todo: function marking categorical type
                    marked_attribute = _attr_val
                    # fp information: if mark_bit = fp_bit xor mask_bit is 1 then change the value, otherwise not
                    if _mark_bit == 1:
                        # selecting attributes for knn search -> this is user specified
                        if _attr_name in self.correlated_attributes:
                            other_attributes = self.correlated_attributes.tolist().copy()
                            other_attributes.remove(_attr_name)
                            bt = balltree[_attr_name]
                        else:
                            other_attributes = self.correlated_attributes.tolist().copy()
                            bt = balltree["all"]
                        if self.distance_based:
                            neighbours, dist = bt.query_radius([relation[other_attributes].loc[r[0]]], r=self.d,
                                                               return_distance=True, sort_results=True)
                        else:
                            # nondeterminism - non chosen tuples with max distance
                            dist, neighbours = bt.query([relation[other_attributes].loc[r[0]]], k=self.k + 1)
                        # excluding the observed tuple
                        neighbours = neighbours[0].tolist()
                        neighbours.remove(neighbours[0])
                        dist = dist[0].tolist()
                        dist.remove(dist[0])
                        # print("Max distance: " + str(max(dist)))
                        # resolve the non-determinism take all the tuples with max distance
                        neighbours, dist = bt.query_radius(
                            [relation[other_attributes].loc[r[0]]], r=max(dist), return_distance=True,
                            sort_results=True)
                        neighbours = neighbours[0].tolist()
                        neighbours.remove(neighbours[0])
                        dist = dist[0].tolist()
                        dist.remove(dist[0])
                        # todo: show this graphically - this is a point for a discussion
                        # print("Size of a neighbourhood: " + str(len(neighbours)) + " instead of " + str(self.k))
                        # print("\tNeighbours: " + str(neighbours))

                        # check the frequencies of the values
                        other_values = []
                        for neighb in neighbours:
                            # force the change of a value if possible
                            if force_change:
                                if relation.at[neighb, r[1].keys()[_attr_idx]] != _attr_val:
                                #if relation.at[neighb, _attr_name] != _attr_val:
                                    other_values.append(relation.at[neighb, r[1].keys()[_attr_idx]])
                                    #other_values.append(relation.at[neighb, _attr_val])
                            else:
                                other_values.append(relation.at[neighb, r[1].keys()[_attr_idx]])

                        frequencies = dict()
                        if len(other_values) != 0:
                            for value in set(other_values):
                                f = other_values.count(value) / len(other_values)
                                frequencies[value] = f
                            if marking_randomness:
                                # choose a value randomly, weighted by a frequency
                                marked_attribute = random.choice(list(frequencies.keys()), 1,
                                                                 p=list(frequencies.values()))[0]
                            else:
                                marked_attribute = max(frequencies.items(), key=operator.itemgetter(1))[0]
                        else:
                            # print("Marking value stays the same.")
                            marked_attribute = _attr_val
                else:  # numerical value
                    # todo: marking integer value
                    # select least significant bit
                    bit_idx = random.randint(0, sys.maxsize) % self.xi
                    # alter the chosen value
                    marked_attribute = set_bit(_attr_val, bit_idx, _mark_bit)
                # print("Index " + str(r[0]) + ", attribute " + str(r[1].keys()[attr_idx]) + ", from " +
                #      str(attribute_val) + " to " + str(marked_attribute))
                #fingerprinted_relation.at[r[0], r[1].keys()[attr_idx]] = marked_attribute
                fingerprinted_relation.at[r[0], _attr_name] = marked_attribute

        # delabeling
        for cat in categorical_attributes:
            fingerprinted_relation[cat] = label_encoders[cat].inverse_transform(fingerprinted_relation[cat])

        print("Fingerprint inserted.")
        if save:
            # todo: fix this
            write_dataset(fingerprinted_relation, "categorical_neighbourhood", dataset, [self.gamma, self.xi],
                          recipient_id)
        print("Time: " + str(int(time.time() - start)) + " sec.")
        return fingerprinted_relation

    def detection(self, dataset=None, secret_key=None, dataset_path=None):
        print("Start blind detection algorithm of fingerprinting scheme for categorical data (neighbourhood)...")
        print("\tgamma: " + str(self.gamma) + "\n\txi: " + str(self.xi))

        # todo: this is the key difference: no original dataset needed

        if secret_key is None:
            secret_key = self.secret_key

        if dataset is None:
            # todo: fix this
            relation_fp, primary_key_fp = import_fingerprinted_dataset(scheme_string="categorical_neighbourhood",
                                                                       dataset_name="blind/" + dataset_path,
                                                                       scheme_params=[self.gamma, self.xi])
        else:
            relation_fp = dataset

        tot_attributes = _count_attributes(relation_fp)
        categorical_attributes = relation_fp.select_dtypes(include='object').columns

        # todo: address checking for the missing columns (defense against vertical attack)
        # if not relation_orig.columns.equals(relation_fp.columns):
        #    print(relation_fp.columns)
        #    difference = relation_orig.columns.difference(relation_fp.columns)
        #    for diff in difference:
        #        relation_fp[diff] = relation_orig[diff]
        # bring back the original order of columns
        # relation_fp = relation_fp[relation_orig.columns.tolist()]

        # encode the categorical values
        label_encoders = dict()
        for cat in categorical_attributes:
            label_enc = LabelEncoder()
            relation_fp[cat] = label_enc.fit_transform(relation_fp[cat])
            label_encoders[cat] = label_enc

        start = time.time()
        balltree = _train_balltrees(relation_fp, self.correlated_attributes)

        votes = [[0, 0] for x in range(self.fingerprint_bit_length)]

        for r in relation_fp.iterrows():
            _data_row_seed(r[0], secret_key)
            # this tuple was marked
            if random.randint(0, sys.maxsize) % self.gamma == 0:
                # this attribute was marked (skip the primary key)
                _attr_idx, _attr_name, _attr_val = _select_marking_attribute(r[1], tot_attributes)
                # fingerprint bit
                fingerprint_idx = random.randint(0, sys.maxsize) % self.fingerprint_bit_length
                # mask
                mask_bit = random.randint(0, sys.maxsize) % 2
                # ######################################################## #
                # ######################################################## #
                # i know this, but i can't know the mark because I don't have the value of a fingerprint at index fp_idx
                # so i need to find out
                # how?
                # check if the value matches the majority class in the neighbourhood
                # problems with this: randomness(if ai have it) and force value change! - changes in insertion needed
                # another problem (harder): the value will likely be the same as majority class regardless the mark

                # if I remove the force change, the results might actually be better
                # if I remove the randomness, the fingerprint might be less robust
                #

                #
                # ######################################################## #
                # ######################################################## #

                # todo: this is different
                # find the neighbourhood
                # sort values by frequencies
                # if the value is the most frequent by those, then the fingerprint bit value was 1, otherwise 0
                # todo: consider this: if there is only one possible attribute in the neighbourhood then the value
                # # # could have been both 0 and 1 equally likely. therefore, give votes to both.
                if _attr_name in categorical_attributes:
                    # selecting attributes for knn search -> this is user specified
                    if _attr_name in self.correlated_attributes:
                        other_attributes = self.correlated_attributes.tolist().copy()
                        other_attributes.remove(_attr_name)
                        bt = balltree[_attr_name]
                    else:
                        other_attributes = self.correlated_attributes.tolist().copy()
                        bt = balltree["all"]
                    if self.distance_based:
                        neighbours, dist = bt.query_radius([relation_fp[other_attributes].loc[r[0]]], r=self.d,
                                                           return_distance=True, sort_results=True)
                    else:
                        # nondeterminism - non chosen tuples with max distance
                        dist, neighbours = bt.query([relation_fp[other_attributes].loc[r[0]]], k=self.k + 1)
                    # excluding the observed tuple - todo: dont exclude
                    neighbours = neighbours[0].tolist()
                    # neighbours.remove(neighbours[0])
                    dist = dist[0].tolist()
                    dist.remove(dist[0])
                    # print("Max distance: " + str(max(dist)))
                    # resolve the non-determinism take all the tuples with max distance
                    neighbours, dist = bt.query_radius(
                        [relation_fp[other_attributes].loc[r[0]]], r=max(dist), return_distance=True,
                        sort_results=True)
                    neighbours = neighbours[0].tolist()
                    neighbours.remove(neighbours[0])
                    dist = dist[0].tolist()
                    dist.remove(dist[0])

                    # check the frequencies of the values
                    possible_values = []
                    for neighb in neighbours:
                        possible_values.append(relation_fp.at[neighb, r[1].keys()[_attr_idx]])
                    frequencies = dict()
                    if len(possible_values) != 0:
                        for value in set(possible_values):
                            f = possible_values.count(value) / len(possible_values)
                            frequencies[value] = f
                        # sort the values by their frequency
                        frequencies = {k: v for k, v in
                                       sorted(frequencies.items(), key=lambda item: item[1], reverse=True)}
                    if _attr_val == list(frequencies.keys())[0]:
                        mark_bit = 1
                    else:
                        mark_bit = 0
                    # original_value = relation_orig.loc[r[0], attr_name]
                    # mark_bit = 0
                    # if attribute_val != original_value:
                    #    mark_bit = 1
                else:
                    bit_idx = random.randint(0, sys.maxsize) % self.xi
                    if _attr_val < 0:
                        _attr_val = -_attr_val
                    mark_bit = (_attr_val >> bit_idx) % 2

                fingerprint_bit = (mark_bit + mask_bit) % 2
                votes[fingerprint_idx][fingerprint_bit] += 1

        # this fingerprint template will be upside-down from the real binary representation
        fingerprint_template = [2] * self.fingerprint_bit_length
        # recover fingerprint
        for i in range(self.fingerprint_bit_length):
            # certainty of a fingerprint value
            T = 0.50
            if votes[i][0] + votes[i][1] != 0:
                if votes[i][0] / (votes[i][0] + votes[i][1]) > T:
                    fingerprint_template[i] = 0
                elif votes[i][1] / (votes[i][0] + votes[i][1]) > T:
                    fingerprint_template[i] = 1

        fingerprint_template_str = ''.join(map(str, fingerprint_template))
        print("Fingerprint detected: " + list_to_string(fingerprint_template))
        print('VOTES')
        print(votes)

        # todo: add probabilistic matching
        recipient_no = super().detect_potential_traitor(fingerprint_template_str, secret_key)
        if recipient_no >= 0:
            print("Recipient " + str(recipient_no) + " is a suspect.")
        else:
            print("None suspected.")
        print("Runtime: " + str(int(time.time() - start)) + " sec.")
        return recipient_no


class NBNNScheme(Scheme):
    """
    Fingerprints numerical and categorical types of data with non-blind Nearest-Neighbors scheme
    """

    __primary_key_len = 20  # supports the data set size of up to 1,048,576 entries

    def __init__(self, gamma=2, xi=2, fingerprint_bit_length=32, secret_key=333, number_of_recipients=10,
                 distance_based=False, d=0, k=10):
        self.gamma = gamma
        self.xi = xi
        self.distance_based = distance_based  # if False, then fixed-size-neighbourhood-based with k=10 - default
        self.correlated_attributes = None
        if distance_based:
            self.d = d
        else:
            self.k = k
        super().__init__(fingerprint_bit_length, secret_key, number_of_recipients)

    def insertion(self, dataset, recipient_id, secret_key=None, save=False):
        init_msg = "Start the insertion algorithm of a scheme for fingerprinting categorical data (neighbourhood) ..." \
                   "\n\tgamma: " + str(self.gamma) + "\n\txi: " + str(self.xi)
        print(init_msg)
        if secret_key is not None:
            self.secret_key = secret_key
        # it is always assumed that the first column in the dataset is the primary key !
        relation, primary_key = import_dataset(dataset)
        tot_attributes = len(relation.columns) - 1

        fingerprint = super().create_fingerprint(recipient_id)
        fp_msg = "\nGenerated fingerprint for recipient " + str(recipient_id) + ": " + fingerprint.bin + "Inserting the " \
                                                                                                 "fingerprint...\n"
        print(fp_msg)

        start = time.time()

        # label encoder for categorical attributes
        categorical_attributes = relation.select_dtypes(include='object').columns
        label_encoders = dict()
        for cat in categorical_attributes:
            label_enc = LabelEncoder()
            relation[cat] = label_enc.fit_transform(relation[cat])
            label_encoders[cat] = label_enc

        # ball trees from user-specified correlated attributes
        self.correlated_attributes = categorical_attributes[:]  # todo: this is a demo set where we consider everything mutually correlated

        balltree = _train_balltrees(relation, self.correlated_attributes)

        fingerprinted_relation = relation.copy()
        for r in relation.iterrows():
            # r[0] is an index of a row = primary key
            # seed = concat(secret_key, primary_key)
            seed = (self.secret_key << self.__primary_key_len) + r[0]
            random.seed(seed)

            # selecting the tuple
            if random.randint(0, sys.maxsize) % self.gamma == 0:
                # selecting the attribute
                attr_idx = random.randint(0, sys.maxsize) % tot_attributes + 1
                attr_name = r[1].index[attr_idx]
                attribute_val = r[1][attr_idx]

                # select fingerprint bit
                fingerprint_idx = random.randint(0, sys.maxsize) % self.fingerprint_bit_length
                fingerprint_bit = fingerprint[fingerprint_idx]
                # select mask and calculate the mark bit
                mask_bit = random.randint(0, sys.maxsize) % 2
                mark_bit = (mask_bit + fingerprint_bit) % 2

                # check if attribute is categorical
                if attr_name in categorical_attributes:
                    marked_attribute = attribute_val
                    # fp information: if mark_bit = fp_bit xor mask_bit is 1 then change the value, otherwise not
                    if mark_bit == 1:
                        # selecting attributes for knn search -> this is user specified
                        if attr_name in self.correlated_attributes:
                            other_attributes = self.correlated_attributes.tolist().copy()
                            other_attributes.remove(attr_name)
                            bt = balltree[attr_name]
                        else:
                            other_attributes = self.correlated_attributes.tolist().copy()
                            bt = balltree["all"]
                        if self.distance_based:
                            neighbours, dist = bt.query_radius([relation[other_attributes].loc[r[0]]], r=self.d,
                                                               return_distance=True, sort_results=True)
                        else:
                            # nondeterminism - non chosen tuples with max distance
                            dist, neighbours = bt.query([relation[other_attributes].loc[r[0]]], k=self.k + 1)
                        # excluding the observed tuple
                        neighbours = neighbours[0].tolist()
                        neighbours.remove(neighbours[0])
                        dist = dist[0].tolist()
                        dist.remove(dist[0])
                        # print("Max distance: " + str(max(dist)))
                        # resolve the non-determinism take all the tuples with max distance
                        neighbours, dist = bt.query_radius(
                            [relation[other_attributes].loc[r[0]]], r=max(dist), return_distance=True,
                            sort_results=True)
                        neighbours = neighbours[0].tolist()
                        neighbours.remove(neighbours[0])
                        dist = dist[0].tolist()
                        dist.remove(dist[0])
                        # todo: show this graphically - this is a point for a discussion
                        # print("Size of a neighbourhood: " + str(len(neighbours)) + " instead of " + str(self.k))
                        # print("\tNeighbours: " + str(neighbours))

                        # check the frequencies of the values
                        other_values = []
                        for neighb in neighbours:
                            # force the change of a value if possible
                            if relation.at[neighb, r[1].keys()[attr_idx]] != attribute_val:
                                other_values.append(relation.at[neighb, r[1].keys()[attr_idx]])
                        frequencies = dict()
                        if len(other_values) != 0:
                            for value in set(other_values):
                                f = other_values.count(value) / len(other_values)
                                frequencies[value] = f
                            # choose a value randomly, weighted by a frequency
                            marked_attribute = random.choice(list(frequencies.keys()), 1,
                                                             p=list(frequencies.values()))[0]
                        else:
                            # print("Marking value stays the same.")
                            marked_attribute = attribute_val
                else:  # numerical value
                    # select least significant bit
                    bit_idx = random.randint(0, sys.maxsize) % self.xi
                    # alter the chosen value
                    marked_attribute = set_bit(attribute_val, bit_idx, mark_bit)
                # print("Index " + str(r[0]) + ", attribute " + str(r[1].keys()[attr_idx]) + ", from " +
                #      str(attribute_val) + " to " + str(marked_attribute))
                fingerprinted_relation.at[r[0], r[1].keys()[attr_idx]] = marked_attribute

        # delabeling
        for cat in categorical_attributes:
            fingerprinted_relation[cat] = label_encoders[cat].inverse_transform(fingerprinted_relation[cat])

        print("Fingerprint inserted.")
        if secret_key is None and save is True:
            write_dataset(fingerprinted_relation, "categorical_neighbourhood", dataset, [self.gamma, self.xi],
                          recipient_id)
        print("Time: " + str(int(time.time() - start)) + " sec.")
        return fingerprinted_relation

    def detection(self, dataset_name, secret_key=None, dataset=None):
        print("Start detection algorithm of fingerprinting scheme for categorical data (neighbourhood)...")
        print("\tgamma: " + str(self.gamma) + "\n\txi: " + str(self.xi))

        relation_orig, primary_key_orig = import_dataset(dataset_name)
        tot_attributes = len(relation_orig.columns) - 1
        categorical_attributes = relation_orig.select_dtypes(include='object').columns

        if secret_key is not None:
            relation_fp = dataset
        else:
            relation_fp, primary_key_fp = import_fingerprinted_dataset(scheme_string="categorical_neighbourhood",
                                                                       dataset_name=dataset_name,
                                                                       scheme_params=[self.gamma, self.xi])
        # check for the missing columns
        if not relation_orig.columns.equals(relation_fp.columns):
            print(relation_fp.columns)
            difference = relation_orig.columns.difference(relation_fp.columns)
            for diff in difference:
                relation_fp[diff] = relation_orig[diff]

        # bring back the original order of columns
        relation_fp = relation_fp[relation_orig.columns.tolist()]

        # encode the categorical values
        label_encoders = dict()
        for cat in categorical_attributes:
            label_enc = LabelEncoder()
            relation_fp[cat] = label_enc.fit_transform(relation_fp[cat])

            label_encoders[cat] = label_enc

        # encode the categorical values of the original
        label_encoders_orig = dict()
        for cat in categorical_attributes:
            label_enc_orig = LabelEncoder()
            relation_orig[cat] = label_enc_orig.fit_transform(relation_orig[cat])
            label_encoders_orig[cat] = label_enc_orig

        start = time.time()

        count = [[0, 0] for x in range(self.fingerprint_bit_length)]

        for r in relation_fp.iterrows():
            seed = (self.secret_key << self.__primary_key_len) + r[0]
            random.seed(seed)
            # this tuple was marked
            if random.randint(0, sys.maxsize) % self.gamma == 0:
                # this attribute was marked (skip the primary key)
                attr_idx = random.randint(0, sys.maxsize) % tot_attributes + 1
                attr_name = r[1].index[attr_idx]
                attribute_val = r[1][attr_idx]
                # fingerprint bit
                fingerprint_idx = random.randint(0, sys.maxsize) % self.fingerprint_bit_length
                # mask
                mask_bit = random.randint(0, sys.maxsize) % 2

                if attr_name in categorical_attributes:
                    original_value = relation_orig.loc[r[0], attr_name]
                    mark_bit = 0
                    if attribute_val != original_value:
                        mark_bit = 1
                else:
                    bit_idx = random.randint(0, sys.maxsize) % self.xi
                    if attribute_val < 0:
                        attribute_val = -attribute_val
                    mark_bit = (attribute_val >> bit_idx) % 2

                fingerprint_bit = (mark_bit + mask_bit) % 2
                count[fingerprint_idx][fingerprint_bit] += 1

        # this fingerprint template will be upside-down from the real binary representation
        fingerprint_template = [2] * self.fingerprint_bit_length
        # recover fingerprint
        for i in range(self.fingerprint_bit_length):
            # certainty of a fingerprint value
            T = 0.50
            if count[i][0] + count[i][1] != 0:
                if count[i][0] / (count[i][0] + count[i][1]) > T:
                    fingerprint_template[i] = 0
                elif count[i][1] / (count[i][0] + count[i][1]) > T:
                    fingerprint_template[i] = 1

        fingerprint_template_str = ''.join(map(str, fingerprint_template))
        print("Fingerprint detected: " + list_to_string(fingerprint_template))

        recipient_no = super().detect_potential_traitor(fingerprint_template_str)
        if recipient_no >= 0:
            print("recipient " + str(recipient_no) + " is a traitor.")
        else:
            print("None suspected.")
        print("Runtime: " + str(int(time.time() - start)) + " sec.")
        return recipient_no


class Universal(Scheme):
    '''
    Fingerprinting scheme applicable to all data types. Unifies the AK scheme for numerical data and adapted AK for
    categorical and decimal data.
        - categorical data: sorted, encoded to numerical and modified with constraints
        - decimal data: TBA
    '''
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

        self._INIT_MESSAGE = "Start insertion algorithm...\n" \
                             "\tgamma: " + str(self.gamma) + "\n\tfingerprint length: " + \
                             str(self.fingerprint_bit_length)

    def insertion(self, dataset, recipient_id, secret_key, save=False, exclude=None, include=None,
                  primary_key_attribute=None, target_attribute=None, write_to=None):
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
        :return: datasets.Dataset instance of fingerprinted data
        '''
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
        count_omega = [0 for i in range(self.fingerprint_bit_length)]
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
                count_omega[fingerprint_idx] += 1
                fingerprint_bit = fingerprint[fingerprint_idx]
                mark_bit = (mask_bit + fingerprint_bit) % 2
                # if the value is categorical, the mark_bit indicates whether the new value will be odd or even
                if fingerprinted_relation.dataframe.columns[attr_idx] in fingerprinted_relation.categorical_attributes:
                    all_vals = fingerprinted_relation.get_distinct(attr_idx)
                    if mark_bit == 1: # odd value
                        possible_vals = [v for v in all_vals if v % 2 == 1]
                    else:
                        possible_vals = [v for v in all_vals if v % 2 == 0]
                    marked_attribute = random.choice(possible_vals)
                else:
                    # alter the chosen value
                    marked_attribute = set_bit(attribute_val, bit_idx, mark_bit)
                fingerprinted_relation.dataframe.at[r[0], r[1].keys()[attr_idx]] = marked_attribute
                count += 1

        # put back the excluded stuff
        fingerprinted_relation = _data_postprocess(fingerprinted_relation, original_data)
        print("Fingerprint inserted.")
        print("\tmarked tuples: ~" + str(round((count / relation.number_of_rows) * 100, 2)) + "%")
        print("\tsingle fingerprint bit embedded " + str(int(np.mean(count_omega))) + " times")
        if save and write_to is None:
            fingerprinted_relation.save("ak_scheme_{}_{}_{}.csv".format(self.gamma, self.fingerprint_bit_length, recipient_id))
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

    def detection(self, dataset, secret_key, exclude=None, include=None, read=False, primary_key_attribute=None,
                  real_recipient_id=None, target_attribute=None):
        print("Start detection algorithm...")
        print("\tgamma: " + str(self.gamma) + "\n\txi: " + str(self.xi))
        fingerprinted_data = _read_data(dataset)
        fingerprinted_data_prep = fingerprinted_data.clone()
        if target_attribute is not None:
            fingerprinted_data_prep._set_target_attribute = target_attribute
        if primary_key_attribute is not None:
            fingerprinted_data_prep._set_primary_key(primary_key_attribute)
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

        recipient_no = super().detect_potential_traitor(fingerprint_template_str, secret_key)
        if recipient_no >= 0:
            print("Recipient " + str(recipient_no) + " is suspected.")
        else:
            print("No one suspected.")
        print("Runtime: " + str(int(time.time() - start)) + " sec.")
        return recipient_no


