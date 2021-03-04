import pandas as pd
import os


class GermanCreditSample:
    def load(self):
        data = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) +
                           '\german_credit_sample.csv')
        return data

    def __init__(self):
        pass


class BreastCancerWisconsin:
    def load(self):
        data = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) +
                           '/breast_cancer_wisconsin.csv')
        return data

    def __init__(self):
        pass


__all__ = ['GermanCreditSample', 'BreastCancerWisconsin']
