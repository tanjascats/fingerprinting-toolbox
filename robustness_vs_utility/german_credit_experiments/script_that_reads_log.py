from ast import *
import ast
import pickle

file = open('robustness_vs_utility/german_credit_experiments/log_formatted(server)', 'r')
contents = file.read()
contents.split('experiment_results_data')
contents = contents.strip()
partitions = contents.split('experiment_results_data')

experiment_results = {'decision tree': [dict(), dict(), dict(), dict()],
                           'knn': [dict(), dict(), dict(), dict()],
                           'logistic regression': [dict(), dict(), dict(), dict()],
                           'svm': [dict(), dict(), dict(), dict()],
                           'gradient boosting': [dict(), dict(), dict(), dict()]}

for p in partitions:
    if len(p) < 2:
        continue
    parameters = p.split(']')
    classifier = parameters[0][2:-1]
    gamma_encoded = int(parameters[1][1:])
    removed = int(parameters[2][1:])
    dictionary_string = parameters[3][3:]
    dictionary = ast.literal_eval(dictionary_string)
    experiment_results[classifier][gamma_encoded][removed] = dictionary

# PICKLE
outfile = open('robustness_vs_utility/german_credit_experiments/experiment_results(server)', 'wb')
pickle.dump(experiment_results, outfile)
outfile.close()
