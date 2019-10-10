import random
import sys
import time
from utils import import_dataset, create_fingerprint, set_bit, write_dataset


class AK:
    gamma = 10  # 1/gamma tuples used in fingerprinting
    xi = 1  # least significant bits
    fingerprint_bit_length = 96  # num_of_tuples/(gamma*fingerprint_length) = 50

    secret_key = 0
    number_of_buyers = 1
    buyer_id = 0

    def __init__(self, gamma, xi, fingerprint_bit_length, secret_key, number_of_buyers, buyer_id):
        self.gamma = gamma
        self.xi = xi
        self.fingerprint_bit_length = fingerprint_bit_length
        self.secret_key = secret_key
        self.number_of_buyers = number_of_buyers
        self.buyer_id = buyer_id

    def insertion(self, dataset_name):
        print("Start AK insertion algorithm...")
        print("\tgamma: " + str(self.gamma) + "\n\txi: " + str(self.xi))
        # it is assumed that the first column in the dataset is the primary key
        relation, primary_key, primary_key_len = import_dataset(dataset_name)
        # number of numerical attributes minus primary key
        num_of_attributes = len(relation.select_dtypes(exclude='object').columns) - 1

        fingerprint = create_fingerprint(self.fingerprint_bit_length, self.buyer_id, self.number_of_buyers,
                                         self.secret_key)
        print("\nGenerated fingerprint for buyer " + str(self.buyer_id) + ": " + fingerprint.bin)
        print("Inserting the fingerprint...\n")

        fingerprinted_relation = relation.copy()
        # count marked tuples
        count = count_omega = 0
        start = time.time()
        for r in relation.select_dtypes(exclude='object').iterrows():
            # seed = concat(secret_key, primary_key)
            seed = (self.secret_key << primary_key_len) + r[1][0]
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
        write_dataset(fingerprinted_relation, "AK", dataset_name, self.gamma, self.xi, self.buyer_id)
        print("Time: " + str(int(time.time() - start)) + " sec.")

# todo: detection
