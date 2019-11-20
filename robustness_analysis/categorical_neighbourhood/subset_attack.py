# this is a script that runs the experiments of subset attack on
# the scheme for categorical data based on a neighbourhood search
# approach
from attacks.subset_attack import SubsetAttack

n_experiments = 1  # number of times we attack the same fingerprinted file
n_fp_experiments = 1  # number of times we run fp insertion

attack = SubsetAttack()

# Workflow:
    # fingerprint data
    # perform a subset attack; perform a few on the same fingerprinted copy
    # try to detect a fingerprint after each attack
    # record success
    # this needs to be repeated >100

for i in range(n_fp_experiments):
    # fingerprint the data

    for j in range(n_experiments):
        # perform the attack
        attack.run()