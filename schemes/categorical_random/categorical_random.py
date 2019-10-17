from schemes.scheme import Scheme


class CategoricalRandom(Scheme):

    def __init__(self, fingerprint_bit_length, secret_key, number_of_buyers):
        super().__init__(fingerprint_bit_length, secret_key, number_of_buyers)

    def insertion(self, dataset_name, buyer_id):
        pass

    def detection(self, dataset_name, real_buyer_id):
        pass