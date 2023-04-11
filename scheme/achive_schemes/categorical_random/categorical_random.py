from scheme.achive_schemes.scheme import Scheme
from utils import*
import time
from sklearn.preprocessing import LabelEncoder
import random
import sys


class CategoricalRandom(Scheme):
    """
    Fingerprints numerical and categorical types of data
    """

    # supports the dataset size of up to 1,048,576 entries
    __primary_key_len = 20

    def __init__(self, gamma, xi, fingerprint_bit_length, secret_key, number_of_buyers):
        self.gamma = gamma
        self.xi = xi
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
        categorical_attributes = relation.select_dtypes(include='object').columns.tolist()
        label_encoders = dict()
        for cat in categorical_attributes:
            label_enc = LabelEncoder()
            relation[cat] = label_enc.fit_transform(relation[cat])
            label_encoders[cat] = label_enc

        fingerprinted_relation = relation.copy()
        # count marked tuples and actual differences
        count = count_omega = diff = 0

        for r in relation.iterrows():
            # r[0] is an index of a row = primary key
            # seed = concat(secret_key, primary_key)
            seed = (self.secret_key << self.__primary_key_len) + r[0]
            random.seed(seed)

            # select the tuple
            if random.randint(0, sys.maxsize) % self.gamma == 0:
                # select attribute (that is not the primary key -> REMOVED!)
                attr_idx = random.randint(0, sys.maxsize) % tot_attributes
                attribute_val = r[1][attr_idx]
                # select least significant bit
                bit_idx = random.randint(0, sys.maxsize) % self.xi
                # select mask bit
                mask_bit = random.randint(0, sys.maxsize) % 2
                # select fingerprint bit
                fingerprint_idx = random.randint(0, sys.maxsize) % self.fingerprint_bit_length
                if fingerprint_idx == 18:
                    count_omega += 1
                fingerprint_bit = fingerprint[fingerprint_idx]
                # fingerprint_temp = fingerprint_temp >> fingerprint_idx
                # fingerprint_bit = fingerprint_temp % 2
                # marking bit = xor(mask bit, fingerprint bit)
                mark_bit = (mask_bit + fingerprint_bit) % 2

                # alter the chosen value
                marked_attribute = set_bit(attribute_val, bit_idx, mark_bit)
                if attribute_val != marked_attribute:
                    diff += 1
                fingerprinted_relation.at[r[0], r[1].keys()[attr_idx]] = marked_attribute
                count += 1
        # todo: decoding categorical attributes

    def detection(self, dataset_name, real_buyer_id):
        pass