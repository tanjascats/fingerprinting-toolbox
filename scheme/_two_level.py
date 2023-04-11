import random
from mpmath import *
from bitstring import BitArray

from utils import *
from utils import _read_data

from ._base import Scheme


class TwoLevelScheme(Scheme):
    '''
    Implements fingerprinting scheme by Guo et al. (2006), aka. Two-level embedding scheme*

    Technical limitations:
        - supports the dataset size of up to 1,048,576 entries
        - applies to data sets with numerical integer values

    * Guo, Fei, Jianmin Wang, and Deyi Li. "Fingerprinting relational databases." Proceedings of the 2006 ACM symposium on Applied computing. 2006.
    '''
    __primary_key_len = 20

    def __init__(self, gamma_1, gamma_2, alpha_1, alpha_2, alpha_3, xi=1, fingerprint_bit_length=None,
                 number_of_recipients=None):
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.xi = xi
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3

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

        self._INIT_MESSAGE = "Two-level scheme - initialised.\nEmbedding started...\nParameters:" \
                             "\tgamma: " + str(self.gamma_1) + "," + str(self.gamma_2) + \
                             "\n\talpha: " + str(self.alpha_1) + ", " + str(self.alpha_2) + ", " + str(self.alpha_3) + \
                             "\n\tfingerprint length: " + str(self.fingerprint_bit_length)

    def insertion(self, dataset, recipient_id, secret_key):
        print("Start Two-level Scheme insertion algorithm...")
        print("\tgamma 1: " + str(self.gamma_1) + "\n\tgamma_2: " + str(self.gamma_2) + "\n\txi: " + str(self.xi) +
              "\n\talpha 1: " + str(self.alpha_1) + "\n\talpha 2: " + str(self.alpha_2) + "\n\talpha 3: "
              + str(self.alpha_3))
        # it is assumed that the first column in the dataset is the primary key
        # relation, primary_key = import_dataset_from_file(dataset)
        relation = _read_data(dataset)
        # number of numerical attributes minus primary key
        num_of_attributes = len(relation.dataframe.select_dtypes(exclude='object').columns) - 1

        fingerprint = super().create_fingerprint(recipient_id, secret_key)
        print("\nGenerated fingerprint for buyer " + str(recipient_id) + ": " + fingerprint.bin)
        print("Inserting the fingerprint...\n")

        fingerprinted_relation = relation.dataframe.copy()
        marked_on_level_1 = 0
        marked_on_level_2 = 0
        marked_on_level_2_arr = []
        subset = [[], []]
        group = [[] for i in range(self.fingerprint_bit_length)]

        start = time.time()
        for r in relation.dataframe.select_dtypes(exclude='object').iterrows():
            # i = hash(pk|k) mod fpt_len
            seed = int((int(r[1][0]) << 16) + secret_key)
            random.seed(seed)
            rand_value = random.getrandbits(100)
            # group
            i = rand_value % self.fingerprint_bit_length  # %8
            # tuple is i-th group
            if int(fingerprint[i]) == 0:  # avoid the same seed as the previous one
                seed = int((((2 << self.__primary_key_len) + r[1][0]) << 16) + secret_key)
            else:
                seed = int((((int(fingerprint[i]) << self.__primary_key_len) + r[1][0]) << 16) + secret_key)
            random.seed(seed)
            rand_value = random.getrandbits(100)
            if rand_value % self.gamma_1 == 0:  # %20
                group[i].append(r[1][0])
                marked_on_level_1 += 1
                subset[int(fingerprint[i])].append(r[1][0])
                # choosing the place for embedding
                seed = int((((r[1][0] << 16) + secret_key) << 1) + int(fingerprint[i]))
                random.seed(seed)
                rand_value = random.getrandbits(100)
                attr_idx = rand_value % num_of_attributes + 1
                attribute_val = r[1][attr_idx]
                j = rand_value % self.xi
                mark_bit = int(fingerprint[i])
                marked_attribute = set_bit(attribute_val, j, mark_bit)
                fingerprinted_relation.at[r[0], r[1].keys()[attr_idx]] = marked_attribute
            else:  # second level
                seed = int((((fingerprint.int << self.__primary_key_len) + int(r[1][0])) << 16) + secret_key)
                random.seed(seed)
                rand_value = random.getrandbits(100)
                if rand_value % self.gamma_2 == 0:
                    marked_on_level_2 += 1
                    marked_on_level_2_arr.append(r[1][0])
                    seed = int((((int(r[1][0]) << 16) + secret_key) << self.fingerprint_bit_length) + fingerprint.int)
                    random.seed(seed)
                    rand_value = random.getrandbits(100)
                    attr_idx = rand_value % num_of_attributes + 1
                    attribute_val = r[1][attr_idx]
                    j = rand_value % self.xi
                    seed = int((secret_key << self.__primary_key_len) + int(r[1][0]))
                    random.seed(seed)
                    rand_value = random.getrandbits(100)
                    marked_attribute = set_bit(attribute_val, j, rand_value % 2)
                    fingerprinted_relation.at[r[0], r[1].keys()[attr_idx]] = marked_attribute

        print("Marked on level 1:  " + str(marked_on_level_1) + " / " + str(len(relation.dataframe)))
        print("Marked on level 2:  " + str(marked_on_level_2) + " / " + str(len(relation.dataframe)))

        print("Fingerprint inserted.")
        # write_dataset(fingerprinted_relation, "two_level_scheme", dataset, [self.gamma_1, self.gamma_2, self.xi], recipient_id)
        print("Time: " + str(int(time.time() - start)) + " sec.")
        return fingerprinted_relation

    def detection(self, dataset, secret_key):
        print("Start Two-level Scheme detection algorithm...")
        print("\tgamma 1: " + str(self.gamma_1) + "\n\tgamma 2: " + str(self.gamma_2) + "\n\txi: " + str(self.xi) +
              "\n\talpha 1: " + str(self.alpha_1) + "\n\talpha 2: " + str(self.alpha_2) + "\n\talpha 3: " + str(self.alpha_3))
        #suspicious_relation, primary_key = import_fingerprinted_dataset(scheme_string="two_level_scheme", dataset_name=dataset,
        #                                                     scheme_params=[self.gamma_1, self.gamma_2, self.xi],
        #                                                     real_buyer_id=real_recipient_id)
        suspicious_relation = _read_data(dataset)
        primary_key = suspicious_relation.primary_key
        start = time.time()

        fingerprint_template = [2] * self.fingerprint_bit_length
        total_count_0, total_count_1, match_count_0, match_count_1 = self.detect(list(primary_key), suspicious_relation,
                                                                                 secret_key)

        total_count = total_count_0 + total_count_1
        match_count = match_count_0 + match_count_1

        print("total count: " + str(total_count))
        print("match count: " + str(match_count))
        print("what should be 75%: " + str(match_count / total_count))

        thr = self.threshold(total_count, self.alpha_1)
        print("threshold: " + str(thr))
        print("threshold/total_count: " + str(thr / total_count))
        if match_count >= self.threshold(total_count, self.alpha_1):
            # pass varification
            print("Ownership verification passed")
            groups = [[] for _ in range(self.fingerprint_bit_length)]
            for tuple_pk in list(primary_key):
                tuple_pk = int(tuple_pk)
                # i = hash(private_k | SECRET_K) mod L
                random.seed((int(tuple_pk) << 16) + secret_key)
                i = random.getrandbits(100) % self.fingerprint_bit_length
                # ith group <- tuple
                groups[i].append(tuple_pk)
            for bit_idx, group in enumerate(groups):
                total_count_0, total_count_1, match_count_0, match_count_1 = self.detect(group, suspicious_relation,
                                                                                         secret_key)
                bit_set = False
                threshold_0 = self.threshold(total_count_0, self.alpha_2)
                if match_count_0 > threshold_0:
                    fingerprint_template[bit_idx] = 0
                    bit_set = True
                threshold_1 = self.threshold(total_count_1, self.alpha_2)
                if match_count_1 > threshold_1:
                    if not bit_set:
                        fingerprint_template[bit_idx] = 1
                    else:
                        if match_count_1 - threshold_1 > match_count_0 - threshold_0:
                            fingerprint_template[bit_idx] = 1
            fingerprint_template_str = ''.join(map(str, fingerprint_template))
            print("Extracted fingerprint: " + str(fingerprint_template_str))
            verified = self.verify_fingerprint(fingerprint_template, primary_key, suspicious_relation, secret_key)
            if verified:
                print("Fingerprint is verified.")
            else:
                print("Fingerprint not verified.")
            traitor = super().detect_potential_traitor(fingerprint_template_str, secret_key)
            if traitor > -1:
                out = "The traitor is buyer " + str(traitor)
                print(out)
            else:
                out = "No one suspected."
                print(out)
        else:
            print("Ownership verification failed.")
            traitor = -1
        print("Runtime: " + str(int(time.time() - start)) + " sec.")
        return traitor

    def detect(self, primary_keys, suspicious_relation, secret_key):
        num_of_attributes = len(suspicious_relation.dataframe.select_dtypes(exclude='object').columns) - 1
        subset = [[], []]
        total_count_0 = total_count_1 = match_count_0 = match_count_1 = 0
        for pk in primary_keys:
            # seed = (((1 << primary_key_len) + pk) << 16) + SECRET_KEY
            seed = (((1 << self.__primary_key_len) + int(pk)) << 16) + secret_key
            random.seed(seed)
            rand_val = random.getrandbits(100)
            if rand_val % self.gamma_1 == 0:
                subset[1].append(pk)
                total_count_1 += 1
                seed = (((int(pk) << 16) + secret_key) << 1) + 1
                random.seed(seed)
                rand_val = random.getrandbits(100)
                attr_idx = rand_val % num_of_attributes + 1
                attribute_val = int(suspicious_relation.dataframe.loc[pk, :][attr_idx])
                j = rand_val % self.xi
                if (attribute_val >> j) % 2 == 1:
                    match_count_1 += 1
            # fingerprint bit is 0 (value is 2 because insertion needed to avoid same seed cases)
            seed = (((2 << self.__primary_key_len) + int(pk)) << 16) + secret_key
            random.seed(seed)
            rand_val = random.getrandbits(100)
            if rand_val % self.gamma_1 == 0:
                subset[0].append(pk)
                total_count_0 += 1
                seed = (((int(pk) << 16) + secret_key) << 1) + 0
                random.seed(seed)
                rand_val = random.getrandbits(100)
                attr_idx = rand_val % num_of_attributes + 1
                attribute_val = int(suspicious_relation.dataframe.loc[pk, :][attr_idx])
                j = rand_val % self.xi
                if (attribute_val >> j) % 2 == 0:
                    match_count_0 += 1
        return total_count_0, total_count_1, match_count_0, match_count_1

    # in first embedding process match_count/total_count is cca 0.75
    def threshold(self, n, alpha):
        # return minimum integer m that satisfies sum(k=m)(n)(binom(n,k)*(0.5)^n) < alpha
        m = n
        sum = power(0.5, n) * binomial(n, m)
        while sum < alpha:
            m -= 1
            sum += power(0.5, n) * binomial(n, m)
        return m + 1

    def verify_fingerprint(self, fingerprint, primary_key, suspicious_relation, secret_key):
        num_of_attributes = len(suspicious_relation.dataframe.select_dtypes(exclude='object').columns) - 1
        # function detect traitor takes array as an input
        fingerprint = BitArray(fingerprint)
        total_count = match_count = 0
        arr = []
        # foreach tuple
        for pk in list(primary_key):
            pk = int(pk)
            # if hash(suspect_fpt|pk|k) mod gamma_2 == 0 &&
            # hash(suspect_fpt[i]|pk|k) mode gamma_1 !=0)) # second level
            seed_0 = int((((fingerprint.int << self.__primary_key_len) + int(pk)) << 16) + secret_key)
            random.seed(seed_0)
            rand_val_0 = random.getrandbits(100)
            seed_1 = (pk << 16) + secret_key
            random.seed(seed_1)
            rand_val_1 = random.getrandbits(100)
            group = rand_val_1 % self.fingerprint_bit_length
            if int(fingerprint[group]) == 0:
                seed_2 = (((2 << self.__primary_key_len) + pk) << 16) + secret_key
            else:
                seed_2 = (((int(fingerprint[group]) << self.__primary_key_len) + int(pk)) << 16) + secret_key
            random.seed(seed_2)
            rand_val_2 = random.getrandbits(100)
            if rand_val_0 % self.gamma_2 == 0 and rand_val_2 % self.gamma_1 != 0:  # if entered the second level embedding
                arr.append(pk)
                # total_count ++
                total_count += 1
                # j = hash(pk|k|suspect_fpt) mod xi
                seed = int((((pk << 16) + secret_key) << self.fingerprint_bit_length) + fingerprint.int)
                random.seed(seed)
                rand_value = random.getrandbits(100)
                attr_idx = rand_value % num_of_attributes + 1
                attribute_val = int(suspicious_relation.dataframe.loc[pk, :][attr_idx])
                j = rand_value % self.xi
                # if hash[k|pk) is even and jth bit is 0:
                seed = (secret_key << self.__primary_key_len) + pk
                random.seed(seed)
                rand_value = random.getrandbits(100)
                if rand_value % 2 == 0 and (attribute_val >> j) % 2 == 0:
                    # match_count++
                    match_count += 1
                # else if hash(k|pk) is odd and jth bit is 1:
                elif rand_value % 2 == 1 and (attribute_val >> j) % 2 == 1:
                    # match_count++
                    match_count += 1
            # if match count > threshold(total-count, alpha3):
        if total_count != 0:
            print("Match count in fingerprint verification phase: " + str(match_count / total_count))
        print("Match count: " + str(match_count))
        if match_count > self.threshold(total_count, self.alpha_3):
            return True  # fingerprint is verified
        else:
            # return false; fingerprint is not verified
            return False
