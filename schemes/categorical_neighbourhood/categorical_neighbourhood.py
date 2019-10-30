from schemes.scheme import Scheme
from utils import *
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.neighbors import BallTree
import numpy.random as random
import sys


class CategoricalNeighbourhood(Scheme):
    """
    Fingerprints numerical and categorical types of data
    """

    # supports the dataset size of up to 1,048,576 entries
    __primary_key_len = 20

    def __init__(self, gamma, xi, fingerprint_bit_length, secret_key, number_of_buyers, distance_based=False, d=0, k=10):
        self.gamma = gamma
        self.xi = xi
        self.distance_based = distance_based  # if False, then fixed-size-neighbourhood-based with k=10 - default
        if distance_based:
            self.d = d
        else:
            self.k = k
        super().__init__(fingerprint_bit_length, secret_key, number_of_buyers)

    def insertion(self, dataset_name, buyer_id):
        print("Start the insertion algorithm of a scheme for fingerprinting categorical data (random) ...")
        print("\tgamma: " + str(self.gamma) + "\n\txi: " + str(self.xi))
        # it is assumed that the first column in the dataset is the primary key
        relation, primary_key = import_dataset(dataset_name)
        # number of numerical attributes minus primary key
        number_of_num_attributes = len(relation.select_dtypes(exclude='object').columns) - 1
        # number of non-numerical attributes
        number_of_cat_attributes = len(relation.select_dtypes(include='object').columns)
        # total number of attributes
        tot_attributes = number_of_num_attributes + number_of_cat_attributes

        fingerprint = super().create_fingerprint(buyer_id)
        print("\nGenerated fingerprint for buyer " + str(buyer_id) + ": " + fingerprint.bin)
        print("Inserting the fingerprint...\n")

        start = time.time()

        # label encoder
        categorical_attributes = relation.select_dtypes(include='object').columns
        label_encoders = dict()
        for cat in categorical_attributes:
            label_enc = LabelEncoder()
            relation[cat] = label_enc.fit_transform(relation[cat])
            label_encoders[cat] = label_enc

        # ball trees from user-specified correlated attributes
        CORRELATED_ATTRIBUTES = categorical_attributes[:-3]  # todo: this is a demo set

        start_training_balltrees = time.time()
        # ball trees from all-except-one attribute and all attributes
        balltree = dict()
        for i in range(len(CORRELATED_ATTRIBUTES)):
            balltree_i = BallTree(relation[CORRELATED_ATTRIBUTES[:i].append(CORRELATED_ATTRIBUTES[(i+1):])],
                                  metric="hamming")
            balltree[CORRELATED_ATTRIBUTES[i]] = balltree_i
        balltree_all = BallTree(relation[CORRELATED_ATTRIBUTES], metric="hamming")
        balltree["all"] = balltree_all
        print("Training balltrees in: " + str(round(time.time() - start_training_balltrees, 2)) + " sec.")

        # algorithm:
        # -choose a tuple
        # -choose an attribute
        # -calculate nearest neighbours within distance 0 regarding other attributes
        # --if this is empty, increase the distance
        # -sort the frequencies of accurences of values and choose the most frequent one (alternatively, random using the sequence generator)
        # ! NONDETERMINISM -> equally frequent values?

        fingerprinted_relation = relation.copy()
        for r in relation.iterrows():
            # r[0] is an index of a row = primary key
            # seed = concat(secret_key, primary_key)
            seed = (self.secret_key << self.__primary_key_len) + r[0]
            random.seed(seed)

            # selecting the tuple
            if random.randint(0, sys.maxsize) % self.gamma == 0:
                # selecting the attribute
                attr_idx = random.randint(0, sys.maxsize) % tot_attributes
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
                        if attr_name in CORRELATED_ATTRIBUTES:
                            other_attributes = CORRELATED_ATTRIBUTES.tolist().copy()
                            other_attributes.remove(attr_name)
                            bt = balltree[attr_name]
                        else:
                            other_attributes = CORRELATED_ATTRIBUTES.tolist().copy()
                            bt = balltree["all"]
                        if self.distance_based:
                            neighbours, dist = bt.query_radius([relation[other_attributes].loc[r[0]]], r=self.d,
                                                               return_distance=True, sort_results=True)
                        else:
                            # nondeterminism - non chosen tuples with max distance
                            dist, neighbours = bt.query([relation[other_attributes].loc[r[0]]], k=self.k+1)
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
                        print("Size of a neighbourhood: " + str(len(neighbours)) + " instead of " + str(self.k))

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
        write_dataset(fingerprinted_relation, "categorical_neighbourhood", dataset_name, [self.gamma, self.xi], buyer_id)
        print("Time: " + str(int(time.time() - start)) + " sec.")
        return True

    def detection(self, dataset_name, real_buyer_id):
        pass
