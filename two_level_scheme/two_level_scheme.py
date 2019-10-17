from scheme import Scheme
from utils import *
import random
import time


class TwoLevelScheme(Scheme):
    # supports the dataset size of up to 1,048,576 entries
    __primary_key_len = 20

    def __init__(self, gamma_1, gamma_2, xi, fingerprint_bit_length, secret_key, number_of_buyers):
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.xi = xi
        super().__init__(fingerprint_bit_length, secret_key, number_of_buyers)

    def insertion(self, dataset_name, buyer_id):
        print("Start Two-level Scheme insertion algorithm...")
        print("\tgamma 1: " + str(self.gamma_1) + "\n\tgamma_2: " + str(self.gamma_2) + "\n\txi: " + str(self.xi))
        # it is assumed that the first column in the dataset is the primary key
        relation, primary_key = import_dataset(dataset_name)
        # number of numerical attributes minus primary key
        num_of_attributes = len(relation.select_dtypes(exclude='object').columns) - 1

        fingerprint = super().create_fingerprint(buyer_id)
        print("\nGenerated fingerprint for buyer " + str(buyer_id) + ": " + fingerprint.bin)
        print("Inserting the fingerprint...\n")

        fingerprinted_relation = relation.copy()
        marked_on_level_1 = 0
        marked_on_level_2 = 0
        marked_on_level_2_arr = []
        # for each tuple
        subset = [[], []]
        group = [[] for i in range(self.fingerprint_bit_length)]

        start = time.time()
        for r in relation.select_dtypes(exclude='object').iterrows():
            # i = hash(pk|k) mod fpt_len
            seed = int((int(r[1][0]) << 16) + self.secret_key)
            random.seed(seed)
            rand_value = random.getrandbits(100)
            # group
            i = rand_value % self.fingerprint_bit_length  # %8
            # tuple is i-th group
            if int(fingerprint[i]) == 0:  # avoid the same seed as the previous one
                seed = int((((2 << self.__primary_key_len) + r[1][0]) << 16) + self.secret_key)
            else:
                seed = int((((int(fingerprint[i]) << self.__primary_key_len) + r[1][0]) << 16) + self.secret_key)
            random.seed(seed)
            rand_value = random.getrandbits(100)
            # print("Primary key: " + str(r[1][0]) + "\n\tfingerprint bit: " + str(int(fingerprint[i])) + "\n\tseed: " + str(seed) + "\n\trandom_val: " + str(rand_value))
            if rand_value % self.gamma_1 == 0:  # %20
                group[i].append(r[1][0])
                marked_on_level_1 += 1
                subset[int(fingerprint[i])].append(r[1][0])
                # print("Passed gamma1")
                # print("Tuple " + str(r[1][0]) + " goes with " + str(int(fingerprint[i])))
                # choosing the place for embedding
                seed = int((((r[1][0] << 16) + self.secret_key) << 1) + int(fingerprint[i]))
                random.seed(seed)
                rand_value = random.getrandbits(100)
                attr_idx = rand_value % num_of_attributes + 1
                attribute_val = r[1][attr_idx]
                # print("Marking the attribute: " + str(attr_idx))
                j = rand_value % self.xi
                # print("Marking the attribute: (" + str(r[1][0]) + " ," + str(attr_idx) + ", " + str(j) + ") - "
                #      + str(int(fingerprint[i])))
                mark_bit = int(fingerprint[i])
                marked_attribute = set_bit(attribute_val, j, mark_bit)
                fingerprinted_relation.at[r[0], r[1].keys()[attr_idx]] = marked_attribute
                # print(" - " + str(attribute_val))
                # print(" - " + str(marked_attribute))
            else:  # second level
                seed = int((((fingerprint.int << self.__primary_key_len) + int(r[1][0])) << 16) + self.secret_key)
                random.seed(seed)
                rand_value = random.getrandbits(100)
                if rand_value % self.gamma_2 == 0:
                    marked_on_level_2 += 1
                    marked_on_level_2_arr.append(r[1][0])
                    seed = int((((int(r[1][0]) << 16) + self.secret_key) << self.fingerprint_bit_length) + fingerprint.int)
                    random.seed(seed)
                    rand_value = random.getrandbits(100)
                    attr_idx = rand_value % num_of_attributes + 1
                    attribute_val = r[1][attr_idx]
                    j = rand_value % self.xi
                    # print("LEVEL 2: Marking the attribute: (" + str(r[1][0]) + " ," + str(attr_idx) + ", " + str(j) + ")")
                    seed = int((self.secret_key << self.__primary_key_len) + int(r[1][0]))
                    random.seed(seed)
                    rand_value = random.getrandbits(100)
                    marked_attribute = set_bit(attribute_val, j, rand_value % 2)
                    fingerprinted_relation.at[r[0], r[1].keys()[attr_idx]] = marked_attribute

        print("Marked on level 1:  " + str(marked_on_level_1))
        print("Marked on level 2:  " + str(marked_on_level_2))
        print("Subset_0: " + str(subset[0]))
        print("Subset_1: " + str(subset[1]))
        # for g in group:
        #    print(g)
        print("Fingerprint inserted.")
        write_dataset(fingerprinted_relation, "two-level_scheme", dataset_name, [self.gamma_1, self.gamma_2, self.xi], buyer_id)
        print("Time: " + str(int(time.time() - start)) + " sec.")

    def detection(self, dataset_name, real_buyer_id):
        pass
