# script for utility evaluation of fingerprinted data
# data statistics before vs after
# ML utility on classification task before vs after
import os
from datasets import *
import pandas as pd


def mean_variance():
    mean_dataframe = pd.DataFrame()
    std_dataframe = pd.DataFrame()
    var_dataframe = pd.DataFrame()
    all_fp_datasets = os.listdir('fingerprinted_data/breast_cancer_w')
    for fp_dataset_path in all_fp_datasets:
        fp_dataset = Dataset(path='fingerprinted_data/breast_cancer_w/' + fp_dataset_path,
                             target_attribute='class', primary_key_attribute='sample-code-number')
        a, b, c, fp_len, gamma, xi, secret_key, r = fp_dataset_path.split('_')
        fp_len = int(fp_len[1:]); gamma = float(gamma[1:]); xi = int(xi[1:])
        if fp_len != 32: continue

        mean = fp_dataset.dataframe.mean()
        std = fp_dataset.dataframe.std()
        var = fp_dataset.dataframe.var()
        mean['gamma'] = var['gamma'] = std['gamma'] = gamma
        mean['xi'] = var['xi'] = std['xi'] = xi

        mean_dataframe = mean_dataframe.append(mean, ignore_index=True)
        std_dataframe = std_dataframe.append(std, ignore_index=True)
        var_dataframe = var_dataframe.append(var, ignore_index=True)

