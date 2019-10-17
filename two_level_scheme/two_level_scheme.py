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
            if rand_value % self.gamma_1 == 0:  # %20
                group[i].append(r[1][0])
                marked_on_level_1 += 1
                subset[int(fingerprint[i])].append(r[1][0])
                # choosing the place for embedding
                seed = int((((r[1][0] << 16) + self.secret_key) << 1) + int(fingerprint[i]))
                random.seed(seed)
                rand_value = random.getrandbits(100)
                attr_idx = rand_value % num_of_attributes + 1
                attribute_val = r[1][attr_idx]
                j = rand_value % self.xi
                mark_bit = int(fingerprint[i])
                marked_attribute = set_bit(attribute_val, j, mark_bit)
                fingerprinted_relation.at[r[0], r[1].keys()[attr_idx]] = marked_attribute
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
                    seed = int((self.secret_key << self.__primary_key_len) + int(r[1][0]))
                    random.seed(seed)
                    rand_value = random.getrandbits(100)
                    marked_attribute = set_bit(attribute_val, j, rand_value % 2)
                    fingerprinted_relation.at[r[0], r[1].keys()[attr_idx]] = marked_attribute

        print("Marked on level 1:  " + str(marked_on_level_1) + " / " + str(len(relation)))
        print("Marked on level 2:  " + str(marked_on_level_2) + " / " + str(len(relation)))

        print("Fingerprint inserted.")
        write_dataset(fingerprinted_relation, "two_level_scheme", dataset_name, [self.gamma_1, self.gamma_2, self.xi], buyer_id)
        print("Time: " + str(int(time.time() - start)) + " sec.")
        return True

    def detection(self, dataset_name, real_buyer_id):
        print("Start Two-level Scheme detection algorithm...")
        print("\tgamma 1: " + str(self.gamma_1) + "\n\tgamma 2: " + str(self.gamma_2) + "\n\txi: " + str(self.xi))
        suspicious_relation, primary_key = import_fingerprinted_dataset(scheme_string="ak_scheme", dataset_name=dataset_name,
                                                             scheme_params=[self.gamma_1, self.gamma_2, self.xi],
                                                             real_buyer_id=real_buyer_id)
        start = time.time()

        # todo

    def detect(self, primary_keys, suspicious_relation):
        num_of_attributes = len(suspicious_relation.select_dtypes(exclude='object').columns) - 1
        subset = [[], []]
        total_count_0 = total_count_1 = match_count_0 = match_count_1 = 0
        for pk in primary_keys:
            # seed = (((1 << primary_key_len) + pk) << 16) + SECRET_KEY
            seed = (((1 << self.__primary_key_len) + int(pk)) << 16) + self.secret_key
            random.seed(seed)
            rand_val = random.getrandbits(100)
            # print("Primary key: " + str(pk) + "\n\tseed: " + str(seed) + "\n\trandom_val: " + str(rand_val))
            # problem here is that the cases where accidentally the values seeded with fp_bit =1 (and are originally 0)
            # will pass the if statement. this is less common for bigger gamma_1, but might happen!!
            if rand_val % self.gamma_1 == 0:
                subset[1].append(pk)
                # print("Subset 1: " + str(pk))
                total_count_1 += 1
                seed = (((int(pk) << 16) + self.secret_key) << 1) + 1
                random.seed(seed)
                rand_val = random.getrandbits(100)
                attr_idx = rand_val % num_of_attributes + 1
                attribute_val = int(suspicious_relation.loc[pk, :][attr_idx])
                # print("Marking the attribute: " + str(attr_idx))
                j = rand_val % self.xi
                # print("Checking the attribute: (" + str(pk) + " ," + str(attr_idx) + ", " + str(j) + ") - 1")
                if (attribute_val >> j) % 2 == 1:
                    # match_count_1 ++
                    # print(" - " + str(attribute_val))
                    match_count_1 += 1
            # fingerprint bit is 0 (value is 2 because insertion needed to avoid same seed cases)
            seed = (((2 << self.__primary_key_len) + int(pk)) << 16) + self.secret_key
            random.seed(seed)
            rand_val = random.getrandbits(100)
            # print("Primary key: " + str(pk) + "\n\tseed: " + str(seed) + "\n\trandom_val: " + str(rand_val))
            if rand_val % self.gamma_1 == 0:
                subset[0].append(pk)
                # print("Subset 0: " + str(pk))
                total_count_0 += 1
                seed = (((int(pk) << 16) + self.secret_key) << 1) + 0
                random.seed(seed)
                rand_val = random.getrandbits(100)
                attr_idx = rand_val % num_of_attributes + 1
                attribute_val = int(suspicious_relation.loc[pk, :][attr_idx])
                # print("Marking the attribute: " + str(attr_idx))
                j = rand_val % self.xi
                # print("Checking the attribute: (" + str(pk) + " ," + str(attr_idx) + ", " + str(j) + ") - 0")
                # print(" - " + str(attribute_val))
                if (attribute_val >> j) % 2 == 0:
                    # match_count_1 ++
                    match_count_0 += 1
        # print("Subset 0 (candidates): " + str(subset[0]))
        # print("Subset 1 (candidates): " + str(subset[1]))
        return total_count_0, total_count_1, match_count_0, match_count_1

