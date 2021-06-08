import random
import pandas as pd
import math
import time
from hashlib import blake2b
from bitstring import BitArray
import sys


#GAMMA = 2
#XI = 2
SECRET_KEY = 3476  #670  #3476  #670
FINGERPRINT_BIT_LENGTH = 64
NUMBER_OF_BUYERS = 10


def main():
    gamma = 3
    filepath = "../fingerprinted_datasets/german_credit_" + str(gamma) + "_2_5.csv"
    # !!! immidiatelly set this after specifying the dataset
    xi = 2
    relation = pd.read_csv(filepath)
    relation = relation.select_dtypes(exclude="object")

    p = 0.0  # percentage of bits to be flipped / probability of choosing a bit for toggling

    # without Id
    num_of_attributes = len(relation.columns) - 1
    num_of_tuples = len(relation)
    available_bits = num_of_tuples * num_of_attributes * xi

    start = time.time()
    sample_size = int(p * available_bits)  # 5,810,120 * p = 5,810

    total_runs = 1
    # here starts the loop
    total_success = 0
    for i in range(total_runs):
        suspect_relation = relation.copy()
        start_iter = time.time()
        sample = random.sample(range(available_bits), sample_size)
        print("Sample chosen. Start flipping...")
        for n in sample:
            #print("Sample: " + str(n))
            tuple_id = int(n / (num_of_attributes * xi))
            #print("Tuple: " + str(tuple_id))
            # skipping Id (+1)
            attr_id = int((n % (num_of_attributes * xi)) / xi) + 1
            #print("Attr: " + str(attr_id))
            lsb_id = int((n % (num_of_attributes * xi)) % xi)
            #print("LSB " + str(lsb_id))

            tuple = suspect_relation.loc[tuple_id, :]
            attr = int(tuple[attr_id])
            #print("Original: " + str(attr))
            # flippin'
            flipped_value = attr ^ 2**lsb_id
            suspect_relation.at[tuple_id, suspect_relation.columns[attr_id]] = flipped_value
            #print("Marked: " + str(suspect_relation.loc[tuple_id, :][attr_id]))

        print("Done flipping: " + str(sample_size) + " bits.")
        print("Time: " + str(int(time.time()-start_iter)) + "sec.")
        suspect_relation.to_csv("flipped_" + str(gamma) + "_" + str(int(p*100)), index=False)
        success, buyer_id, runtime = detection(suspect_relation, gamma, xi)
        if success:
            # TODO checking if the correct buyer is accused should be checked
            total_success += 1

    total_attack_success_rate = 1 - total_success / total_runs
    total_runtime = int((start-time.time())/60)  # in minutes

    result_file = "gamma_" + str(gamma) + "_p_" + str(int(p*100)) + ".txt"
    with open(result_file, "a+") as f:
        f.write("Statistics for fingerprinted data (by AK)\n")
        f.write("File: " + filepath + "\n")
        f.write("gamma = " + str(gamma) + ", xi = " + str(xi) + ", p = " + str(int(p*100)) + "\n")
        f.write("Total runs: " + str(total_runs) + "\n\n")
        f.write("->Results:\n")
        f.write("Fingerprints detected: " + str(total_success) + "/" + str(total_runs) + "\n")
        f.write("Attack success rate: " + str(total_attack_success_rate) + "\n\n")
        f.write("Total runtime: " + str(total_runtime) + "min.\n")

    print("Done with experiment in: " + str(total_runtime) + "min.")
    print("Results in " + result_file)


# determine a traitor
# parameter fingerprint: string representation of bit list
def detect_traitor(fingerprint, secret_key, fingerprint_length, number_of_buyers):
    shift = 10
    # for each buyer
    for buyer_id in range(number_of_buyers):
        buyer_seed = (secret_key << shift) + buyer_id
        b = blake2b(key=buyer_seed.to_bytes(6, 'little'), digest_size=int(FINGERPRINT_BIT_LENGTH / 8))
        buyer_fp = BitArray(hex=b.hexdigest())
        buyer_fp = buyer_fp.bin
        if buyer_fp == fingerprint:
            return buyer_id
    return -1


def detection(suspect_relation, gamma, xi):
    success = False
    start = time.time()
    # number of numerical attributes minus primary key
    num_of_attributes = len(suspect_relation.columns) - 1
    # number of tuples
    num_of_tuples = len(suspect_relation.select_dtypes(exclude='object'))
    # detect primary key
    primary_key = suspect_relation[suspect_relation.columns[0]]
    #print(primary_key)
    # bit range for encoding the primary key
    primary_key_len = math.floor(math.log(max(primary_key), 2)) + 1
    #print(primary_key_len)
    # init fingerprint template and counts
    # for each of the fingerprint bit the votes if it is 0 or 1
    count = [[0, 0] for x in range(FINGERPRINT_BIT_LENGTH)]

    cnt = 0
    # scan all tuples and obtain counts for each fingerprint bit
    for r in suspect_relation.iterrows():
        # seed = concat(secret_key, primary_key)
        seed = (SECRET_KEY << primary_key_len) + r[1][0]
        random.seed(seed)

        # this tuple was marked
        if random.randint(0, sys.maxsize) % gamma == 0:
            # this attribute was marked (skip the primary key)
            attr_idx = random.randint(0, sys.maxsize) % num_of_attributes + 1
            attribute_val = r[1][attr_idx]
            # this LS bit was marked
            bit_idx = random.randint(0, sys.maxsize) % xi
            # if the LSB doesn't exist(?) then skip to the next tuple
            # marked bit is the bit at position bit_idx
                # bit_idx last bit of attribute val
            # take care of negative values
            if attribute_val < 0:
                attribute_val = -attribute_val
                # raise flag
            mark_bit = (attribute_val >> bit_idx) % 2
            mask_bit = random.randint(0, sys.maxsize) % 2
            # fingerprint bit = mark_bit xor mask_bit
            fingerprint_bit = (mark_bit + mask_bit) % 2
            fingerprint_idx = random.randint(0, sys.maxsize) % FINGERPRINT_BIT_LENGTH
            # update votes
            count[fingerprint_idx][fingerprint_bit] += 1

    # this fingerprint template will be upside-down from the real binary representation
    fingerprint_template = [2] * FINGERPRINT_BIT_LENGTH
    # recover fingerprint
    for i in range(FINGERPRINT_BIT_LENGTH):
        if count[i][0] + count[i][1] == 0:
            print("1. None suspected")
            runtime = time.time() - start
            print("Runtime: " + str(int(runtime)) + " sec.")
            return success, -1, runtime
        # certainty of a fingerprint value
        T = 0.50
        if count[i][0] > count[i][1] and count[i][0]/(count[i][0] + count[i][1]) > T:
            fingerprint_template[i] = 0
        elif count[i][1] > count[i][0] and count[i][1]/(count[i][0] + count[i][1]) > T:
            fingerprint_template[i] = 1
        else:
            print("2. None suspected")
            runtime = time.time() - start
            return success, -1, runtime

    fingerprint_template_str = ''.join(map(str, fingerprint_template))
    print("Fingerprint detected: " + list_to_string(fingerprint_template))

    buyer_no = detect_traitor(fingerprint_template_str, SECRET_KEY, FINGERPRINT_BIT_LENGTH, NUMBER_OF_BUYERS)
    if buyer_no >= 0:
        print("Buyer " + str(buyer_no) + " is a traitor.")
        success = True
    else:
        print("None suspected.")
    runtime = int(time.time() - start)
    print("Runtime: " + str(runtime) + " sec.")
    return success, buyer_no, runtime


# returns string representation of list of bits
def list_to_string(l):
    s = ""
    for el in l:
        s += str(el)
    return s


if __name__ == '__main__':
    main()
