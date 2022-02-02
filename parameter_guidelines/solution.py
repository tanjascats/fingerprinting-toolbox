import pandas as pd
import pickle
import time


def parse_solutions_from_raw_data():
    dir = 'parameter_guidelines/evaluation/german_credit/solution/'
    solution = pd.DataFrame(columns=['%rows_marked',  # multi-index
                                     '#columns_marked',  # multi-index
                                     'robustness_horizontal',
                                     'robustness_vertical',
                                     'robustness_flipping',
                                     'utility_loss_best',
                                     'utility_loss_2nd_best'
                                     ])
    solution = solution.set_index(['%rows_marked', '#columns_marked'])

    for property_name in solution.columns:
        property_f = dir + '_{}.pickle'.format(property_name)
        with open(property_f, 'rb') as infile:
            prop = pickle.load(infile)
        for p_rows in prop.index:
            for col in prop.columns:
                n_cols = int(col.split('_')[-1])
                solution.at[(p_rows, n_cols), property_name] = prop.at[p_rows, col]

    with open(dir + 'solution_space_{}.pickle'.format(int(time.time())), 'wb') as outfile:
        pickle.dump(solution, outfile)
    solution.to_csv(dir + 'solution_space_{}.csv'.format(int(time.time())))


if __name__ == '__main__':
    parse_solutions_from_raw_data()
