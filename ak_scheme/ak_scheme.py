from scheme import Scheme
from utils import *
import sys
import random
import time


class AKScheme(Scheme):

    # supports the dataset size of up to 1,048,576 entries
    __primary_key_len = 20

    def __init__(self, gamma, xi, fingerprint_bit_length, secret_key, number_of_buyers):
        self.gamma = gamma
        self.xi = xi
        super().__init__(fingerprint_bit_length, secret_key, number_of_buyers)

    def insertion(self, dataset_name, buyer_id):
        print("Start AK insertion algorithm...")
        print("\tgamma: " + str(self.gamma) + "\n\txi: " + str(self.xi))
        # it is assumed that the first column in the dataset is the primary key
        relation, primary_key = import_dataset(dataset_name)
        # number of numerical attributes minus primary key
        num_of_attributes = len(relation.select_dtypes(exclude='object').columns) - 1

        fingerprint = super().create_fingerprint(buyer_id)
        print("\nGenerated fingerprint for buyer " + str(buyer_id) + ": " + fingerprint.bin)
        print("Inserting the fingerprint...\n")

        fingerprinted_relation = relation.copy()
        # count marked tuples
        count = count_omega = 0
        start = time.time()
        for r in relation.select_dtypes(exclude='object').iterrows():
            # seed = concat(secret_key, primary_key)
            seed = (self.secret_key << self.__primary_key_len) + r[1][0]
            random.seed(seed)

            # select the tuple
            if random.randint(0, sys.maxsize) % self.gamma == 0:
                # select attribute (that is not the primary key)
                attr_idx = random.randint(0, sys.maxsize) % num_of_attributes + 1
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

        print("Fingerprint inserted.")
        print("\tmarked tuples: ~" + str((count / len(relation)) * 100) + "%")
        print("\tsingle fingerprint bit embedded " + str(count_omega) + " times")
        write_dataset(fingerprinted_relation, "AK", dataset_name, self.gamma, self.xi, buyer_id)
        print("Time: " + str(int(time.time() - start)) + " sec.")

    def detection(self, dataset_name, real_buyer_id):
        print("Start AK detection algorithm...")
        print("\tgamma: " + str(self.gamma) + "\n\txi: " + str(self.xi))
        relation, primary_key = import_fingerprinted_dataset(scheme_string="AK", dataset_name=dataset_name,
                                                             gamma=self.gamma, xi=self.xi, real_buyer_id=real_buyer_id)
        start = time.time()
        # number of numerical attributes minus primary key
        num_of_attributes = len(relation.select_dtypes(exclude='object').columns) - 1

        # init fingerprint template and counts
        # for each of the fingerprint bit the votes if it is 0 or 1
        count = [[0, 0] for x in range(self.fingerprint_bit_length)]

        # scan all tuples and obtain counts for each fingerprint bit
        for r in relation.select_dtypes(exclude='object').iterrows():
            seed = (self.secret_key << self.__primary_key_len) + r[1][0]
            random.seed(seed)

            # this tuple was marked
            if random.randint(0, sys.maxsize) % self.gamma == 0:
                # this attribute was marked (skip the primary key)
                attr_idx = random.randint(0, sys.maxsize) % num_of_attributes + 1
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
            if count[i][0]/(count[i][0] + count[i][1]) > T:
                fingerprint_template[i] = 0
            elif count[i][1]/(count[i][0] + count[i][1]) > T:
                fingerprint_template[i] = 1

        fingerprint_template_str = ''.join(map(str, fingerprint_template))
        print("Fingerprint detected: " + list_to_string(fingerprint_template))

        buyer_no = super().detect_potential_traitor(fingerprint_template_str)
        if buyer_no >= 0:
            print("Buyer " + str(buyer_no) + " is a traitor.")
        else:
            print("None suspected.")
        print("Runtime: " + str(int(time.time() - start)) + " sec.")
        return buyer_no