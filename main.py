from AK.AK import AK

scheme = AK(gamma=5, xi=1, fingerprint_bit_length=16, number_of_buyers=10, buyer_id=0, secret_key=333)
#scheme.insertion("covtype_data_int_sample")
scheme.detection("covtype_data_int")