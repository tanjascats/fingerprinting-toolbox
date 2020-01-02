# this is a script that runs the experiments of a bit-flipping attack on
# the scheme for categorical data based on a neighbourhood search
# approach
import random
from datetime import datetime
import numpy as np

from attacks.bit_flipping_attack import BitFlippingAttack
from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood

n_experiments = 10  # (20) number of times we attack the same fingerprinted file
n_fp_experiments = 25  # (50) number of times we run fp insertion

fractions = np.array([0.01, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
results = []
gamma = 30; xi = 1; fingerprint_bit_length = 64

scheme = CategoricalNeighbourhood(gamma=gamma, xi=xi, fingerprint_bit_length=fingerprint_bit_length)
attack = BitFlippingAttack()
data = "nursery"

for size in fractions:

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
            # try to extract the fingerprint
            suspect = scheme.detection(dataset_name=data, real_buyer_id=1, secret_key=secret_key,
                                dataset=release_data)
            if suspect == 1:
                correct += 1
            elif suspect != -1:
                    misdiagnosis += 1

    # write to log file
    f = open("robustness_analysis/categorical_neighbourhood/log/bit_flipping_attack_" + data + ".txt", "a+")
    f.write(str(datetime.fromtimestamp(int(datetime.timestamp(datetime.now())))))
    f.write("\nseed: " + str(seed))
    f.write("\nData: " + data)
    f.write("\n(fraction flipped, gamma, xi, length of a fingerprint): " + str((size, gamma, xi,
                                                                            fingerprint_bit_length)))
    f.write("\nCorrect: " + str(correct) + "/" + str(n_experiments*n_fp_experiments))
    f.write("\nWrong: " + str(n_experiments*n_fp_experiments - correct) + "/" + str(n_experiments*n_fp_experiments)
          + "\n\t- out of which misdiagnosed: " + str(misdiagnosis))
    f.write("\n\n--------------------------------------------------------------\n\n")

    results.append(correct)
    f.write("intermediate summary\n")
    f.write("\n(gamma, xi, length of a fingerprint): " + str((gamma, xi, fingerprint_bit_length)))
    f.write("\nCorrect: " + str(results) + "\n\t/" + str(n_experiments * n_fp_experiments))
    f.close()

f = open("robustness_analysis/categorical_neighbourhood/log/bit_flipping_attack_" + data + ".txt", "a+")
f.write("SUMMARY\n")
f.write(str(datetime.fromtimestamp(int(datetime.timestamp(datetime.now())))))
f.write("\n(gamma, xi, length of a fingerprint): " + str((gamma, xi, fingerprint_bit_length)))
f.write("\nCorrect: " + str(results) + "\n\t/" + str(n_experiments * n_fp_experiments))
f.write("\n\n--------------------------------------------------------------\n\n")
f.close()

print("SUMMARY\n")
print(str(datetime.fromtimestamp(int(datetime.timestamp(datetime.now())))))
print("\n(gamma, xi, length of a fingerprint): " + str((gamma, xi, fingerprint_bit_length)))
print("\nCorrect: " + str(results) + "\n\t/" + str(n_experiments * n_fp_experiments))

