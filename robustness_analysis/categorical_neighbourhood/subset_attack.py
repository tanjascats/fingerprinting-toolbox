# this is a script that runs the experiments of subset attack on
# the scheme for categorical data based on a neighbourhood search
# approach
import random

from attacks.subset_attack import SubsetAttack
from schemes.categorical_neighbourhood.categorical_neighbourhood import CategoricalNeighbourhood

n_experiments = 100  # number of times we attack the same fingerprinted file
n_fp_experiments = 1  # number of times we run fp insertion

size_of_subset = 0.4
gamma = 3; xi = 2

scheme = CategoricalNeighbourhood(gamma=gamma, xi=xi)
attack = SubsetAttack()

# for reproducibility
random.seed(333)

correct, misdiagnosis = 0, 0
for i in range(n_fp_experiments):
    # fingerprint the data
    secret_key = random.randint(0, 1000)
    fp_dataset = scheme.insertion(dataset_name='german_credit', buyer_id=1, secret_key=secret_key)

    for j in range(n_experiments):
        # perform the attack
        release_data = attack.run(dataset=fp_dataset, fraction=size_of_subset)
        # try to extract the fingerprint
        suspect = scheme.detection(dataset_name='german_credit', real_buyer_id=1, secret_key=secret_key,
                            dataset=release_data)
        if suspect == 1:
            correct += 1
        elif suspect != -1:
                misdiagnosis += 1

print("\n\n--------------------------------------------------------------\n\n"
      "Correct: " + str(correct) + "/" + str(n_experiments*n_fp_experiments))
print("Wrong: " + str(n_experiments*n_fp_experiments - correct) + "/" + str(n_experiments*n_fp_experiments)
      + "\n\t- out of which misdiagnosed: " + str(misdiagnosis))

# todo write to log files