# this is a script that runs the experiments of subset attack on
# the scheme for categorical data based on a neighbourhood search
# approach
import sys
sys.path.append("/home/sarcevic/fingerprinting-toolbox/")

import random
from datetime import datetime
import numpy as np
import pandas as pd
import os

from attacks.horizontal_subset_attack import HorizontalSubsetAttack
from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood

from sklearn import preprocessing

from sdv import SDV
from sdv import Metadata


def synth(data, amount):
    le = preprocessing.LabelEncoder()

    #synthetic_data = os.path.join(file_path + '/', "SDV_syntraindata.csv")
    #input_df = pd.read_csv(data, skipinitialspace=True)
    input_df = data
    tables = {'ftable': input_df}

    #trainjson = os.path.join(file_path, "meta.json")
    #instance = os.path.join(file_path, "sdv.pkl")

    metadata = Metadata()
    metadata.add_table('ftable', data=tables['ftable'])
    #metadata.to_json(trainjson)

    sdv = SDV()
    sdv.fit(metadata, tables)
    #sdv.save(instance)

    #sdv = SDV.load(instance)
    samples = sdv.sample_all(amount)  # here: number of synthetic samples
    df = samples['ftable']
    df = df[input_df.columns]
    #df.to_csv(synthetic_data, index=False)
    return df


n_experiments = 8  # number of times we attack the same fingerprinted file
n_fp_experiments = 15  # number of times we run fp insertion

size_of_subset = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
                           0.8, 0.85, 0.9, 0.95, 1])
results = []
gamma = 5; xi = 2; fingerprint_bit_length = 16

scheme = CategoricalNeighbourhood(gamma=gamma, xi=xi, fingerprint_bit_length=fingerprint_bit_length)
attack = HorizontalSubsetAttack()
data = 'nursery'

f = open("robustness_analysis/categorical_neighbourhood/log/horizontal_with_imputation_" + data + ".txt", "a+")

for size in size_of_subset:
    # for reproducibility
    seed = 332
    random.seed(seed)

    correct, misdiagnosis = 0, 0
    for i in range(n_fp_experiments):
        # fingerprint the data
        secret_key = random.randint(0, 1000)
        fp_dataset = scheme.insertion(dataset_name=data, buyer_id=1, secret_key=secret_key)

        for j in range(n_experiments):
            # perform the attack
            release_data = attack.run(dataset=fp_dataset, fraction=size)
            # todo: here goes the imputation
            synthetic_amount = len(fp_dataset) - len(release_data)
            synthetic_data = synth(fp_dataset, synthetic_amount)
            # todo: solve the Id-s
            # get the id-s that are missing release data
            missing_ids = pd.concat([fp_dataset['Id'], release_data['Id']]).drop_duplicates(keep=False)
            synthetic_data = synthetic_data.assign(Id=missing_ids.values)
            imputed_data = release_data.append(synthetic_data)
            print(len(release_data))
            # try to extract the fingerprint
            suspect = scheme.detection(dataset_name=data, real_buyer_id=1, secret_key=secret_key,
                                dataset=imputed_data)
            if suspect == 1:
                correct += 1
            elif suspect != -1:
                    misdiagnosis += 1

    print("\n\n--------------------------------------------------------------\n\n")
    print("Data: " + data)
    print("(size of subset, gamma, xi, length of a fingerprint): " + str((size, gamma, xi, fingerprint_bit_length)))
    print("Correct: " + str(correct) + "/" + str(n_experiments*n_fp_experiments))
    print("Wrong: " + str(n_experiments*n_fp_experiments - correct) + "/" + str(n_experiments*n_fp_experiments)
          + "\n\t- out of which misdiagnosed: " + str(misdiagnosis))

    # write to log file
    f.write(str(datetime.fromtimestamp(int(datetime.timestamp(datetime.now())))))
    f.write("\nseed: " + str(seed))
    f.write("\nData: " + data)
    f.write("\n(size of subset, gamma, xi, length of a fingerprint): " + str((size, gamma, xi,
                                                                            fingerprint_bit_length)))
    f.write("\nCorrect: " + str(correct) + "/" + str(n_experiments*n_fp_experiments))
    f.write("\nWrong: " + str(n_experiments*n_fp_experiments - correct) + "/" + str(n_experiments*n_fp_experiments)
          + "\n\t- out of which misdiagnosed: " + str(misdiagnosis))
    f.write("\n\n--------------------------------------------------------------\n\n")

    results.append(correct)

f.write("SUMMARY\n")
f.write(str(datetime.fromtimestamp(int(datetime.timestamp(datetime.now())))))
f.write("\n(gamma, xi, length of a fingerprint): " + str((gamma, xi, fingerprint_bit_length)))
f.write("\nCorrect: " + str(results) + "\n\t/" + str(n_experiments * n_fp_experiments))
f.write("\n\n--------------------------------------------------------------\n\n")
f.close()
