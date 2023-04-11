import unittest

from scheme import TwoLevelScheme


class TestTwoLevel(unittest.TestCase):
    def test_init(self):
        scheme = TwoLevelScheme(gamma_1=1, gamma_2=2, alpha_1=1, alpha_2=2, alpha_3=3)
        self.assertIsNotNone(scheme)

    def test_detection_integer_data_1(self):
        scheme = TwoLevelScheme(gamma_1=2, gamma_2=2, alpha_1=0.9, alpha_2=0.8, alpha_3=0.8)
        secret_key = 123
        recipient = 0
        data = '../../datasets/cover_type_sample.csv'
        fingerprinted = scheme.insertion(data, recipient, secret_key)
        suspect = scheme.detection(fingerprinted, secret_key)
        self.assertEqual(suspect, recipient)


if __name__ == '__main__':
    unittest.main()
