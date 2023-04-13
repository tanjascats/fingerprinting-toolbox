import numpy.random as random
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import BallTree

from utils import *
from utils import _read_data

from ._base import Scheme

_MAXINT = 2**31 - 1


class CategoricalNeighbourhood(Scheme):
    """
    Fingerprinting scheme proposed by Sarcevic et al. (2020)*.
    Fingerprints numerical and categorical types of data via nearest-neighbourhood search to preserve semantic coherence
    of the attributes.

    * Sarcevic, Tanja, and Rudolf Mayer. "A Correlation-Preserving Fingerprinting Technique for Categorical Data in Relational Databases." ICT Systems Security and Privacy Protection: 35th IFIP TC 11 International Conference, SEC 2020, Maribor, Slovenia, September 21–23, 2020, Proceedings 35. Springer International Publishing, 2020.
    """

    # supports the dataset size of up to 1,048,576 entries
    __primary_key_len = 20

    def __init__(self, gamma, xi=1, fingerprint_bit_length=None, number_of_recipients=None, distance_based=False,
                 d=0, k=10):
        self.gamma = gamma
        self.xi = xi
        self.distance_based = distance_based  # if False, then fixed-size-neighbourhood-based with k=10 - default
        self.correlated_attributes = None
        if distance_based:
            self.d = d
        else:
            self.k = k

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

        self._INIT_MESSAGE = "kNN-based fingerprinting scheme - initialised.\nEmbedding started...\n" \
                             "\tgamma: " + str(self.gamma) + "\n\tfingerprint length: " + \
                             str(self.fingerprint_bit_length) + "\n\tdistance based: " + str(self.distance_based)

    def insertion(self, dataset_name, recipient_id, secret_key):
        print("Start the insertion algorithm of a scheme for fingerprinting categorical data (neighbourhood) ...")
        print("\tgamma: " + str(self.gamma) + "\n\txi: " + str(self.xi))
        if secret_key is not None:
            self.secret_key = secret_key
        # it is assumed that the first column in the dataset is the primary key
        relation, primary_key = import_dataset_from_file(dataset_name)
        # number of numerical attributes minus primary key
        number_of_num_attributes = len(relation.select_dtypes(exclude='object').columns) - 1
        # number of non-numerical attributes
        number_of_cat_attributes = len(relation.select_dtypes(include='object').columns)
        # total number of attributes
        tot_attributes = number_of_num_attributes + number_of_cat_attributes

        fingerprint = super().create_fingerprint(recipient_id, secret_key)
        print("\nGenerated fingerprint for buyer " + str(recipient_id) + ": " + fingerprint.bin)
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
        self.correlated_attributes = categorical_attributes[:]  # todo: this is a demo set

        start_training_balltrees = time.time()
        # ball trees from all-except-one attribute and all attributes
        balltree = dict()
        for i in range(len(self.correlated_attributes)):
            balltree_i = BallTree(relation[self.correlated_attributes[:i].append(self.correlated_attributes[(i + 1):])],
                                  metric="hamming")
            balltree[self.correlated_attributes[i]] = balltree_i
        balltree_all = BallTree(relation[self.correlated_attributes], metric="hamming")
        balltree["all"] = balltree_all
        print("Training balltrees in: " + str(round(time.time() - start_training_balltrees, 2)) + " sec.")

        fingerprinted_relation = relation.copy()
        for r in relation.iterrows():
            # r[0] is an index of a row = primary key
            # seed = concat(secret_key, primary_key)
            seed = (self.secret_key << self.__primary_key_len) + r[0]
            random.seed(seed)

            # selecting the tuple
            if random.randint(0, _MAXINT) % self.gamma == 0:
                # selecting the attribute
                attr_idx = random.randint(0, _MAXINT) % tot_attributes + 1
                attr_name = r[1].index[attr_idx]
                attribute_val = r[1][attr_idx]

                # select fingerprint bit
                fingerprint_idx = random.randint(0, _MAXINT) % self.fingerprint_bit_length
                fingerprint_bit = fingerprint[fingerprint_idx]
                # select mask and calculate the mark bit
                mask_bit = random.randint(0, _MAXINT) % 2
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
                    bit_idx = random.randint(0, _MAXINT) % self.xi
                    # alter the chosen value
                    marked_attribute = set_bit(attribute_val, bit_idx, mark_bit)
                # print("Index " + str(r[0]) + ", attribute " + str(r[1].keys()[attr_idx]) + ", from " +
                #      str(attribute_val) + " to " + str(marked_attribute))
                fingerprinted_relation.at[r[0], r[1].keys()[attr_idx]] = marked_attribute

        # delabeling
        for cat in categorical_attributes:
            fingerprinted_relation[cat] = label_encoders[cat].inverse_transform(fingerprinted_relation[cat])

        print("Fingerprint inserted.")
        if secret_key is None:
            write_dataset(fingerprinted_relation, "categorical_neighbourhood", dataset_name, [self.gamma, self.xi],
                          recipient_id)
        print("Time: " + str(int(time.time() - start)) + " sec.")
        return fingerprinted_relation

    def detection(self, dataset, secret_key, original_data=None):
        print("Start detection algorithm of fingerprinting scheme for categorical data (neighbourhood)...")
        print("\tgamma: " + str(self.gamma) + "\n\txi: " + str(self.xi))

        # relation_orig, primary_key_orig = import_dataset_from_file(dataset)
        relation_orig, primary_key = import_dataset_from_file(original_data)
        relation_fp = _read_data(dataset)
        # number of numerical attributes minus primary key
        number_of_num_attributes = len(relation_orig.select_dtypes(exclude='object').columns) - 1
        number_of_cat_attributes = len(relation_orig.select_dtypes(include='object').columns)
        tot_attributes = number_of_num_attributes + number_of_cat_attributes
        categorical_attributes = relation_orig.select_dtypes(include='object').columns

        #if secret_key is not None:
        #    relation_fp = dataset
        #else:
        #    relation_fp, primary_key_fp = import_fingerprinted_dataset(scheme_string="categorical_neighbourhood",
        #                                                               dataset_name=dataset,
        #                                                               scheme_params=[self.gamma, self.xi],
        #                                                               real_recipient_id=real_recipient_id)

        # check for the missing columns
        if not relation_orig.columns.equals(relation_fp.columns):
            print(relation_fp.columns)
            difference = relation_orig.columns.difference(relation_fp.columns)
            for diff in difference:
                relation_fp[diff] = relation_orig[diff]

        # bring back the original order of columns
        relation_fp.dataframe = relation_fp.dataframe[relation_orig.columns.tolist()]

        # encode the categorical values
        label_encoders = dict()
        for cat in categorical_attributes:
            label_enc = LabelEncoder()
            relation_fp.dataframe[cat] = label_enc.fit_transform(relation_fp.dataframe[cat])

            label_encoders[cat] = label_enc

        # encode the categorical values of the original
        label_encoders_orig = dict()
        for cat in categorical_attributes:
            label_enc_orig = LabelEncoder()
            relation_orig[cat] = label_enc_orig.fit_transform(relation_orig[cat])
            label_encoders_orig[cat] = label_enc_orig

        start = time.time()

        count = [[0, 0] for x in range(self.fingerprint_bit_length)]

        for r in relation_fp.dataframe.iterrows():
            seed = (self.secret_key << self.__primary_key_len) + r[0]
            random.seed(seed)
            # this tuple was marked
            if random.randint(0, _MAXINT) % self.gamma == 0:
                # this attribute was marked (skip the primary key)
                attr_idx = random.randint(0, _MAXINT) % tot_attributes + 1
                attr_name = r[1].index[attr_idx]
                attribute_val = r[1][attr_idx]
                # fingerprint bit
                fingerprint_idx = random.randint(0, _MAXINT) % self.fingerprint_bit_length
                # mask
                mask_bit = random.randint(0, _MAXINT) % 2

                if attr_name in categorical_attributes:
                    original_value = relation_orig.loc[r[0], attr_name]
                    mark_bit = 0
                    if attribute_val != original_value:
                        mark_bit = 1
                else:
                    bit_idx = random.randint(0, _MAXINT) % self.xi
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

        buyer_no = super().detect_potential_traitor(fingerprint_template_str, secret_key)
        if buyer_no >= 0:
            print("Buyer " + str(buyer_no) + " is a traitor.")
        else:
            print("None suspected.")
        print("Runtime: " + str(int(time.time() - start)) + " sec.")
        return buyer_no

    def blind_insertion(self, dataset_name, recipient_id, secret_key=None):
        # todo: version still in testing phase
        # all the same with exception:
        # choose the value in the same way
        # if the fp bit is 1 then change it to the most common in the neighbourhood
        # otherwise change it to the second most common (include self or not - donnow - it would for sure bring less
        # # # alterations - probably a good idea actually)
        print("Start the blind insertion algorithm of a scheme for fingerprinting categorical data (neighbourhood) ...")
        print("\tgamma: " + str(self.gamma) + "\n\txi: " + str(self.xi))
        if secret_key is not None:
            self.secret_key = secret_key
        # it is assumed that the first column in the dataset is the primary key
        relation, primary_key = import_dataset(dataset_name)
        # number of numerical attributes minus primary key
        number_of_num_attributes = len(relation.select_dtypes(exclude='object').columns) - 1
        # number of non-numerical attributes
        number_of_cat_attributes = len(relation.select_dtypes(include='object').columns)
        # total number of attributes
        tot_attributes = number_of_num_attributes + number_of_cat_attributes

        fingerprint = super().create_fingerprint(recipient_id)
        print("\nGenerated fingerprint for buyer " + str(recipient_id) + ": " + fingerprint.bin)
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
        self.correlated_attributes = categorical_attributes[:]  # todo: this is a demo set

        start_training_balltrees = time.time()
        # ball trees from all-except-one attribute and all attributes
        balltree = dict()
        for i in range(len(self.correlated_attributes)):
            balltree_i = BallTree(relation[self.correlated_attributes[:i].append(self.correlated_attributes[(i + 1):])],
                                  metric="hamming")
            balltree[self.correlated_attributes[i]] = balltree_i
        balltree_all = BallTree(relation[self.correlated_attributes], metric="hamming")
        balltree["all"] = balltree_all
        print("Training balltrees in: " + str(round(time.time() - start_training_balltrees, 2)) + " sec.")

        fingerprinted_relation = relation.copy()
        for r in relation.iterrows():
            # r[0] is an index of a row = primary key
            # seed = concat(secret_key, primary_key)
            seed = (self.secret_key << self.__primary_key_len) + r[0]
            random.seed(seed)

            # selecting the tuple
            if random.randint(0, _MAXINT) % self.gamma == 0:
                # selecting the attribute
                attr_idx = random.randint(0, _MAXINT) % tot_attributes
                attr_name = r[1].index[attr_idx]
                attribute_val = r[1][attr_idx]

                # select fingerprint bit
                fingerprint_idx = random.randint(0, _MAXINT) % self.fingerprint_bit_length
                fingerprint_bit = fingerprint[fingerprint_idx]
                # select mask and calculate the mark bit
                mask_bit = random.randint(0, _MAXINT) % 2
                mark_bit = (mask_bit + fingerprint_bit) % 2

                # check if attribute is categorical
                if attr_name in categorical_attributes:
                    marked_attribute = attribute_val
                    # fp information: if mark_bit = fp_bit xor mask_bit is 1 then take the most frequent value,
                    # # # otherwise the second most frequent

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
                    # excluding the observed tuple - todo: maybe i want to keep the observed one since I am not forcing the change
                    neighbours = neighbours[0].tolist()
                    # neighbours.remove(neighbours[0])
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
                    possible_values = []
                    for neighb in neighbours:
                        # force the change of a value if possible
                        # todo: this is different! no need to force the change
                        possible_values.append(relation.at[neighb, r[1].keys()[attr_idx]])
                    frequencies = dict()
                    if len(possible_values) != 0:
                        for value in set(possible_values):
                            f = possible_values.count(value) / len(possible_values)
                            frequencies[value] = f
                        # choose a value randomly, weighted by a frequency
                        # todo: this is different - choose a value based on the fingerprint bit value
                        # sort the values by their frequency
                        frequencies = {k: v for k, v in sorted(frequencies.items(), key=lambda item: item[1], reverse=True)}
                        if mark_bit == 0 and len(frequencies.keys()) > 1:
                            marked_attribute = list(frequencies.keys())[1]
                        else:
                            marked_attribute = list(frequencies.keys())[0]

                else:  # numerical value
                    # select least significant bit
                    bit_idx = random.randint(0, _MAXINT) % self.xi
                    # alter the chosen value
                    marked_attribute = set_bit(attribute_val, bit_idx, mark_bit)
                # print("Index " + str(r[0]) + ", attribute " + str(r[1].keys()[attr_idx]) + ", from " +
                #      str(attribute_val) + " to " + str(marked_attribute))
                fingerprinted_relation.at[r[0], r[1].keys()[attr_idx]] = marked_attribute

        # delabeling
        for cat in categorical_attributes:
            fingerprinted_relation[cat] = label_encoders[cat].inverse_transform(fingerprinted_relation[cat])

        print("Fingerprint inserted.")
        if secret_key is None:
            write_dataset(fingerprinted_relation, "categorical_neighbourhood", "blind/" + dataset_name, [self.gamma, self.xi],
                          recipient_id)
        print("Time: " + str(int(time.time() - start)) + " sec.")
        return fingerprinted_relation

    def blind_detection(self, dataset_name, real_recipient_id, secret_key=None, dataset=None):
        print("Start blind detection algorithm of fingerprinting scheme for categorical data (neighbourhood)...")
        print("\tgamma: " + str(self.gamma) + "\n\txi: " + str(self.xi))

        # todo: this is the key difference: no original dataset needed

        if secret_key is not None:
            relation_fp = dataset
        else:
            relation_fp, primary_key_fp = import_fingerprinted_dataset(scheme_string="categorical_neighbourhood",
                                                                       dataset_name="blind/" + dataset_name,
                                                                       scheme_params=[self.gamma, self.xi],
                                                                       real_recipient_id=real_recipient_id)
        # number of numerical attributes minus primary key
        number_of_num_attributes = len(relation_fp.select_dtypes(exclude='object').columns) - 1
        number_of_cat_attributes = len(relation_fp.select_dtypes(include='object').columns)
        tot_attributes = number_of_num_attributes + number_of_cat_attributes
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

        start_balling = time.time()
        # ball trees from all-except-one attribute and all attributes
        balltree = dict()
        for i in range(len(self.correlated_attributes)):
            balltree_i = BallTree(relation_fp[self.correlated_attributes[:i].append(self.correlated_attributes[(i + 1):])],
                                  metric="hamming")
            balltree[self.correlated_attributes[i]] = balltree_i
        balltree_all = BallTree(relation_fp[self.correlated_attributes], metric="hamming")
        balltree["all"] = balltree_all

        count = [[0, 0] for x in range(self.fingerprint_bit_length)]

        for r in relation_fp.iterrows():
            seed = (self.secret_key << self.__primary_key_len) + r[0]
            random.seed(seed)
            # this tuple was marked
            if random.randint(0, _MAXINT) % self.gamma == 0:
                # this attribute was marked (skip the primary key)
                attr_idx = random.randint(0, _MAXINT) % tot_attributes
                attr_name = r[1].index[attr_idx]
                attribute_val = r[1][attr_idx]
                # fingerprint bit
                fingerprint_idx = random.randint(0, _MAXINT) % self.fingerprint_bit_length
                # mask
                mask_bit = random.randint(0, _MAXINT) % 2

                # todo: this is different
                # find the neighbourhood
                # sort values by frequencies
                # if the value is the most frequent by those, then the fingerprint bit value was 1, otherwise 0
                # todo: consider this: if there is only one possible attribute in the neighbourhood then the value
                # # # could have been both 0 and 1 equally likely. therefore, give votes to both.
                if attr_name in categorical_attributes:
                    # selecting attributes for knn search -> this is user specified
                    if attr_name in self.correlated_attributes:
                        other_attributes = self.correlated_attributes.tolist().copy()
                        other_attributes.remove(attr_name)
                        bt = balltree[attr_name]
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
                        possible_values.append(relation_fp.at[neighb, r[1].keys()[attr_idx]])
                    frequencies = dict()
                    if len(possible_values) != 0:
                        for value in set(possible_values):
                            f = possible_values.count(value) / len(possible_values)
                            frequencies[value] = f
                        # sort the values by their frequency
                        frequencies = {k: v for k, v in
                                       sorted(frequencies.items(), key=lambda item: item[1], reverse=True)}
                    if attribute_val == list(frequencies.keys())[0]:
                        mark_bit = 1
                    else:
                        mark_bit = 0
                    # original_value = relation_orig.loc[r[0], attr_name]
                    # mark_bit = 0
                    # if attribute_val != original_value:
                    #    mark_bit = 1
                else:
                    bit_idx = random.randint(0, _MAXINT) % self.xi
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
        print(count)

        buyer_no = super().detect_potential_traitor(fingerprint_template_str)
        if buyer_no >= 0:
            print("Buyer " + str(buyer_no) + " is a traitor.")
        else:
            print("None suspected.")
        print("Runtime: " + str(int(time.time() - start)) + " sec.")
        return buyer_no
