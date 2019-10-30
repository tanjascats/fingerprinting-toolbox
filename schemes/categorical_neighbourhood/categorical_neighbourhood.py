from schemes.scheme import Scheme
from utils import *
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.neighbors import BallTree


class CategoricalNeighbourhood(Scheme):
    """
    Fingerprints numerical and categorical types of data
    """

    # supports the dataset size of up to 1,048,576 entries
    __primary_key_len = 20

    def __init__(self, gamma, xi, fingerprint_bit_length, secret_key, number_of_buyers):
        self.gamma = gamma
        self.xi = xi
        super().__init__(fingerprint_bit_length, secret_key, number_of_buyers)

    def insertion(self, dataset_name, buyer_id):
        print("Start the insertion algorithm of a scheme for fingerprinting categorical data (random) ...")
        print("\tgamma: " + str(self.gamma) + "\n\txi: " + str(self.xi))
        # it is assumed that the first column in the dataset is the primary key
        relation, primary_key = import_dataset(dataset_name)
        # number of numerical attributes minus primary key
        number_of_num_attributes = len(relation.select_dtypes(exclude='object').columns) - 1
        # number of non-numerical attributes
        number_of_cat_attributes = len(relation.select_dtypes(include='object').columns)
        # total number of attributes
        tot_attributes = number_of_num_attributes + number_of_cat_attributes

        fingerprint = super().create_fingerprint(buyer_id)
        print("\nGenerated fingerprint for buyer " + str(buyer_id) + ": " + fingerprint.bin)
        print("Inserting the fingerprint...\n")

        start = time.time()

        # label encoder
        categorical_attributes = relation.select_dtypes(include='object').columns.tolist()
        label_encoders = dict()
        for cat in categorical_attributes:
            label_enc = LabelEncoder()
            relation[cat] = label_enc.fit_transform(relation[cat])
            label_encoders[cat] = label_enc

        # ball trees from user-specified correlated attributes
        CORRELATED_ATTRIBUTES = categorical_attributes[:-3]

        start_training_balltrees = time.time()
        # ball trees from all-except-one attribute and all attributes
        balltree = dict()
        for i in range(len(CORRELATED_ATTRIBUTES)):
            balltree_i = BallTree(relation[CORRELATED_ATTRIBUTES[:i].append(CORRELATED_ATTRIBUTES[(i + 1):])],
                                  metric="hamming")
            balltree[CORRELATED_ATTRIBUTES[i]] = balltree_i
        balltree_all = BallTree(relation[CORRELATED_ATTRIBUTES], metric="hamming")
        balltree["all"] = balltree_all
        print("Training balltrees in: " + str(round(time.time() - start_training_balltrees, 2)) + " sec.")

    def detection(self, dataset_name, real_buyer_id):
        pass
