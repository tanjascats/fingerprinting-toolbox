from scheme import AKScheme
from attacks.horizontal_subset_attack import HorizontalSubsetAttack


def main():
    scheme = AKScheme(gamma=10, xi=1, fingerprint_bit_length=96, number_of_buyers=10,
                      secret_key=333)
    # fingerprint embedding phase
    fingerprinted_dataset = scheme.insertion(dataset_name="covtype_data_int", buyer_id=0)

    # attack
    attack = HorizontalSubsetAttack()
    altered_dataset = attack.run(dataset=fingerprinted_dataset, fraction=0.10)

    # fingerprint detection phase
    scheme.detection(altered_dataset, real_buyer_id=0)


if __name__ == '__main__':
    main()
