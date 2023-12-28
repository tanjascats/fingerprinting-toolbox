import os
import pandas as pd

__all__ = [
    "read_fingerprinted_data"
]


# todo: *args all together
def read_fingerprinted_data(data_name,
                            attr_subset=None, scheme=None, gamma=None, xi=None, fp_len=None, user_id=None, secret_key=None):
    fingerprinted_data_properties = {'attr_subset': attr_subset,
                                     'scheme': scheme,
                                     'gamma': gamma,
                                     'xi': xi,
                                     'fp_len': fp_len,
                                     'user_id': user_id,
                                     'secret_key': secret_key}
    DATA_PATH = 'parameter_guidelines/fingerprinted_data/'
    fp_data = None
    if os.path.isdir(DATA_PATH+data_name):
        DATA_PATH+=data_name
        if all(value is None for value in fingerprinted_data_properties.values()):
            pass
            # todo: return the set of fp files that match given properties
        else:
            for value in fingerprinted_data_properties.values():
                if value is not None:
                    file_path = '/attr_subset_{}/{}_g{}_x{}_l{}_u{}_sk{}.csv'.format(attr_subset, scheme, gamma, xi,
                                                                                    fp_len, user_id, secret_key)
                    fp_data = pd.read_csv(DATA_PATH+file_path)

    else:
        print('ERROR: fingerprinted data "{}" does not exist.'.format(data_name))
    return fp_data
