# this is a script that runs the experiments of subset attack on
# the scheme for categorical data based on a neighbourhood search
# approach
import sys
sys.path.append("/home/sarcevic/fingerprinting-toolbox/")

import random
from datetime import datetime
import numpy as np

from attacks.combined_attack import CombinedAttack
from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood
import itertools

n_experiments = 1 #6 #10  # number of times we attack the same fingerprinted file
n_fp_experiments = 1 #15 #25  # number of times we run fp insertion

size_of_subset = np.array([0.9, 0.95, 1])
fractions = np.array([0.9, 0.95, 1])
columns = np.array([1, 2])
a = [size_of_subset, columns, fractions]
combinations = list(itertools.product(*a))
results = []
gamma = 3; xi = 2; fingerprint_bit_length = 8

scheme = CategoricalNeighbourhood(gamma=gamma, xi=xi, fingerprint_bit_length=fingerprint_bit_length)
attack = CombinedAttack()
data = 'german_credit'

f = open("robustness_analysis/categorical_neighbourhood/log/combined_attack_" + data + ".txt", "a+")

for combo in combinations:
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
            release_data = attack.run(dataset=fp_dataset, fraction_subset=combo[0], number_of_columns=combo[1],
                                      fraction_flipping=combo[2])
            # try to extract the fingerprint
            suspect = scheme.detection(dataset_name=data, real_buyer_id=1, secret_key=secret_key,
                                dataset=release_data)
            if suspect == 1:
                correct += 1
            elif suspect != -1:
                    misdiagnosis += 1

    print("\n\n--------------------------------------------------------------\n\n")
    print("Data: " + data)
    print("(size of subset, columns, flipped, gamma, xi, length of a fingerprint): " +
          str((combo[0], combo[1], combo[2], gamma, xi, fingerprint_bit_length)))
    print("Correct: " + str(correct) + "/" + str(n_experiments*n_fp_experiments))
    print("Wrong: " + str(n_experiments*n_fp_experiments - correct) + "/" + str(n_experiments*n_fp_experiments)
          + "\n\t- out of which misdiagnosed: " + str(misdiagnosis))

    # write to log file
    f.write(str(datetime.fromtimestamp(int(datetime.timestamp(datetime.now())))))
    f.write("\nseed: " + str(seed))
    f.write("\nData: " + data)
    f.write("\n(size of subset, columns, flipped, gamma, xi, length of a fingerprint): " + str((combo[0], combo[1], combo[2], gamma, xi,
                                                                            fingerprint_bit_length)))
    f.write("\nCorrect: " + str(correct) + "/" + str(n_experiments*n_fp_experiments))
    f.write("\nWrong: " + str(n_experiments*n_fp_experiments - correct) + "/" + str(n_experiments*n_fp_experiments)
          + "\n\t- out of which misdiagnosed: " + str(misdiagnosis))
    f.write("\n\n--------------------------------------------------------------\n\n")

    results.append(correct)

f.write("SUMMARY\n")
f.write(str(datetime.fromtimestamp(int(datetime.timestamp(datetime.now())))))
f.write("\n(gamma, xi, length of a fingerprint): " + str((gamma, xi, fingerprint_bit_length)))
f.write("\n" + str(combinations))
f.write("\nCorrect: " + str(results) + "\n\t/" + str(n_experiments * n_fp_experiments))
f.write("\n\n--------------------------------------------------------------\n\n")
f.close()
