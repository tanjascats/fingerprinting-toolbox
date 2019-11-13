import pandas as pd
import math


# returns the pandas structure of the dataset and its primary key
"""
:returns relation, primary key
"""
def import_dataset(dataset_name):
    filepath = "datasets/" + dataset_name + ".csv"
    relation = pd.read_csv(filepath)
    print("Dataset: " + filepath)

    # detect primary key
    primary_key = relation[relation.columns[0]]
    return relation, primary_key


def import_fingerprinted_dataset(scheme_string, dataset_name, scheme_params, real_buyer_id):
    params_string = ""
    for param in scheme_params:
        params_string += str(param) + "_"
    filepath = "schemes/" + scheme_string + "/fingerprinted_datasets/" + dataset_name + "_" + params_string + \
               str(real_buyer_id) + ".csv"
    relation = pd.read_csv(filepath)
    print("Dataset: " + filepath)

    # detect primary key
    primary_key = relation[relation.columns[0]]
    return relation, primary_key


# sets an idx-th bit of val to mark and returns the new value
def set_bit(val, idx, mark):
    # number of bits necessary for binary representation of val
    neg_val = False
    if val < 0:
        neg_val = True
        val = -val
    if val == 0:
        mask_len = 1
    else:
        mask_len = math.floor(math.log(val, 2)) + 1
    mask = 0
    for i in range(0, mask_len):
        if i != idx:
            mask += 2 ** i
    val = val & mask
    if mark:
        val += 2 ** idx
    if neg_val:
        val = -val
    return val


def write_dataset(fingerprinted_relation, scheme_string, dataset_name, scheme_params, buyer_id):
    params_string = ""
    for param in scheme_params:
        params_string += str(param) + "_"
    new_path = "schemes/" + scheme_string + "/fingerprinted_datasets/" + \
               dataset_name + "_" + params_string + str(buyer_id) + ".csv"
    fingerprinted_relation.to_csv(new_path, index=False)
    print("\tfingerprinted dataset written to: " + new_path)


def list_to_string(l):
    s = ""
    for el in l:
        s += str(el)
    return s