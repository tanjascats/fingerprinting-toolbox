import pandas as pd
import math
import os
import numpy as np
import time
import numbers
import warnings
from traceback import format_exc
from astropy.table import Table
from joblib import Parallel

from sklearn.utils.validation import _check_fit_params, _num_samples
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils import indexable
from sklearn.utils.fixes import delayed
from sklearn.base import clone, is_classifier
from sklearn.model_selection._validation import _score, _aggregate_score_dicts, _normalize_score_results, _insert_error_scores
from sklearn.model_selection._split import check_cv
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics._scorer import check_scoring, _check_multimetric_scoring

from datasets import Dataset

# returns the pandas structure of the dataset and its primary key
def import_dataset(dataset_name):
    """
    :returns relation, primary key
    """
    print("(utils.import_dataset) Warning! Method is deprecated and will be replaced. Please use import_dataset_from_file")
    filepath = "../../datasets/" + str(dataset_name) + ".csv"
    if os.path.isfile(filepath):
        relation = pd.read_csv(filepath)
    else:
        print('dataset not found in datasets directory')
        relation = dataset_name

    print("Dataset: " + filepath)

    # detect primary key
    primary_key = relation[relation.columns[0]]
    return relation, primary_key


def import_dataset_from_file(data_path, primary_key_attribute=None):
    dataset = pd.read_csv(data_path)
    if primary_key_attribute is not None:
        primary_key = dataset[primary_key_attribute]
        dataset.drop(primary_key_attribute)
    else:
        primary_key = dataset.index
    # todo: check if the primary_key is unique
    return dataset, primary_key


def import_fingerprinted_dataset(scheme_string, dataset_name, scheme_params, real_buyer_id=None):
    if real_buyer_id is None:
        relation = dataset_name
    else:
        params_string = ""
        for param in scheme_params:
            params_string += str(param) + "_"
        filepath = "achive_schemes/" + scheme_string + "/fingerprinted_datasets/" + dataset_name + "_" + params_string + \
                   str(real_buyer_id) + ".csv"
        relation = pd.read_csv(filepath)
        print("Dataset: " + filepath)

    # detect primary key
    primary_key = relation[relation.columns[0]]
    return relation, primary_key


# sets an idx-th bit of val to mark and returns the new value
def set_bit(val, idx, mark):
    # number of bits necessary for binary representation of val
    neg_val = False
    if val < 0:
        neg_val = True
        val = -val
    if val == 0:
        mask_len = 1
    else:
        mask_len = math.floor(math.log(val, 2)) + 1
    mask = 0
    for i in range(0, mask_len):
        if i != idx:
            mask += 2 ** i
    val = val & mask
    if mark:
        val += 2 ** idx
    if neg_val:
        val = -val
    return val


def write_dataset(fingerprinted_relation, scheme_string, dataset_name, scheme_params, buyer_id):
    params_string = ""
    for param in scheme_params:
        params_string += str(param) + "_"
    new_path = "achive_schemes/" + scheme_string + "/fingerprinted_datasets/" + \
               dataset_name + "_" + params_string + str(buyer_id) + ".csv"
    fingerprinted_relation.to_csv(new_path, index=False)
    print("\tfingerprinted dataset written to: " + new_path)


def list_to_string(l):
    s = ""
    for el in l:
        s += str(el)
    return s


def count_differences(dataset1, dataset2):
    if len(dataset1) != len(dataset2):
        print("Please pass two datasets of same size.")
    # todo


def _read_data(dataset, primary_key_attribute=None, target_attribute=None):
    '''
    Creates the instance of Dataset for given data.
    :param dataset: string, pandas dataframe or Dataset
    :param primary_key_attribute: name of the primary key attribute
    :param target_attribute: name of the target attribute
    :return: Dataset instance
    '''
    relation = None
    if isinstance(dataset, str):  # assumed the path is given
        relation = Dataset(path=dataset, target_attribute=target_attribute,
                           primary_key_attribute=primary_key_attribute)
    elif isinstance(dataset, pd.DataFrame):  # assumed the pd.DataFrame is given
        relation = Dataset(dataframe=dataset, target_attribute=target_attribute,
                           primary_key_attribute=primary_key_attribute)
    elif isinstance(dataset, Dataset):
        relation = dataset
    else:
        print('Wrong type of input data.')
        exit()
    return relation


def read_data_with_target(dataset_name, scheme_name=None, params=None, buyer_id=None):
    if scheme_name is None:
        data = pd.read_csv("datasets/" + dataset_name + ".csv")
    else:
        params_string = ""
        for param in params:
            params_string += str(param) + "_"
        data = pd.read_csv("achive_schemes/" + scheme_name + "/fingerprinted_datasets/" + dataset_name +
                           "_" + params_string + str(buyer_id) + ".csv")
    target_file = pd.read_csv("datasets/_" + dataset_name + ".csv")
    data["target"] = target_file["target"]
    return data


def add_target(dataset, dataset_name):
    data = dataset
    target_file = pd.read_csv("datasets/_" + dataset_name + ".csv")
    dataset["target"] = target_file["target"]
    return data


# customized fit_and_score method for evaluating fingerprinted data
def fp_fit_and_score(estimator, X_original, y_original, X_fingerprinted, y_fingerprinted, scorer, train_original,
                     test_original, train_fingerprinted, test_fingerprinted, verbose,
                     parameters, fit_params, return_train_score=False,
                     return_parameters=False, return_n_test_samples=False,
                     return_times=False, return_estimator=False,
                     split_progress=None, candidate_progress=None,
                     error_score=np.nan):
    if not isinstance(error_score, numbers.Number) and error_score != 'raise':
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += (f"; {candidate_progress[0]+1}/"
                             f"{candidate_progress[1]}")

    if verbose > 1:
        if parameters is None:
            params_msg = ''
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = (', '.join(f'{k}={parameters[k]}'
                                    for k in sorted_keys))
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X_fingerprinted, fit_params, train_fingerprinted)

    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.time()

    # here I need to make sure to split the fingerprinted data IN THE SAME WAY
    # original train data should be unused
    # fingerprinted test data should be unused
    X_train_original, y_train_original = _safe_split(estimator, X_original, y_original, train_original)
    X_test_original, y_test_original = _safe_split(estimator, X_original, y_original, test_original, train_original)

    X_train_fingerprinted, y_train_fingerprinted = _safe_split(estimator, X_fingerprinted, y_fingerprinted, train_fingerprinted)
    X_test_fingerprinted, y_test_fingerprinted = _safe_split(estimator, X_fingerprinted, y_fingerprinted, test_fingerprinted, train_fingerprinted)

    result = {}
    # fit the model on FINGERPRINTED data
    try:
        if y_train_fingerprinted is None:
            estimator.fit(X_train_fingerprinted, **fit_params)
        else:
            estimator.fit(X_train_fingerprinted, y_train_fingerprinted, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
            warnings.warn("Estimator fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%s" %
                          (error_score, format_exc()),
                          FitFailedWarning)
        result["fit_failed"] = True
    else:
        result["fit_failed"] = False

        fit_time = time.time() - start_time
        # obtain test scores from testing ORIGINAL test data against ORIGINAL target
        test_scores = _score(estimator, X_test_original, y_test_original, scorer, error_score)
        # VERIFICATION PRINTOUTS
        # print(len(X_train_original.index))
        # print(len(X_train_fingerprinted.index))
        # print(type(X_test_original.index))
        # print(X_train_original.index)
        # print(X_train_fingerprinted.index)
        # print(X_train_original.index.equals(X_train_fingerprinted.index))
        # print(X_train_original.columns[1])
        # print(X_train_original[X_train_original.columns[1]].compare
        #       (X_train_fingerprinted[X_train_fingerprinted.columns[1]]))
        # print('----------------')
        # print(X_test_original[X_test_original.columns[1]].compare(X_test_fingerprinted[X_test_fingerprinted.columns[1]]))
        # print('________________')
        # print('Target should look the same')
        # print(y_train_fingerprinted.compare(y_train_original))
        # print('________________')
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            # train scores are based on FINGERPRINTED data
            train_scores = _score(
                estimator, X_train_fingerprinted, y_train_fingerprinted, scorer, error_score
            )

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2 and isinstance(test_scores, dict):
            for scorer_name in sorted(test_scores):
                result_msg += f" {scorer_name}: ("
                if return_train_score:
                    scorer_scores = train_scores[scorer_name]
                    result_msg += f"train={scorer_scores:.3f}, "
                result_msg += f"test={test_scores[scorer_name]:.3f})"
        result_msg += f" total time={logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    result["test_scores"] = test_scores
    if return_train_score:
        result["train_scores"] = train_scores
    if return_n_test_samples:
        result["n_test_samples"] = _num_samples(X_test_original)
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
    return result


def fp_cross_val_score(estimator, X_original, y_original, X_fingerprint, y_fingerprint, cv=5, scoring=None, n_jobs=None,
                       verbose=0, pre_dispatch='2*n_jobs', groups=None, fit_params=None,  return_train_score=False,
                       return_estimator=False, error_score=np.nan):
    '''
    Perform a custom cross validation on fingerprinted data such that the model is trained on fingerprinted, but
    evaluated on original data
    Beware that the X_original, y_original, X_fingerprint and y_fingerprint are expected to match on index!
    There is no index matching within this method.
    '''
    X_original, y_original = indexable(X_original, y_original)

    cv = check_cv(cv, y_original, classifier=is_classifier(estimator))

    if callable(scoring):
        scorers = scoring
    elif scoring is None or isinstance(scoring, str):
        scorers = check_scoring(estimator, scoring)
    else:
        scorers = _check_multimetric_scoring(estimator, scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    results = parallel(
        delayed(fp_fit_and_score)(
            clone(estimator), X_original, y_original, X_fingerprint, y_fingerprint, scorers, train_original,
            test_original, train_fingerprint, test_fingerprint, verbose, None,
            fit_params, return_train_score=return_train_score,
            return_times=True, return_estimator=return_estimator,
            error_score=error_score)
        for (train_original, test_original), (train_fingerprint, test_fingerprint)
        in zip(cv.split(X_original, y_original, groups), cv.split(X_fingerprint, y_fingerprint, groups)))
    # issues might be above. Check this step

    # For callabe scoring, the return type is only know after calling. If the
    # return type is a dictionary, the error scores can now be inserted with
    # the correct key.
    if callable(scoring):
        _insert_error_scores(results, error_score)

    results = _aggregate_score_dicts(results)

    ret = {}
    ret['fit_time'] = results["fit_time"]
    ret['score_time'] = results["score_time"]

    if return_estimator:
        ret['estimator'] = results["estimator"]

    test_scores_dict = _normalize_score_results(results["test_scores"])
    if return_train_score:
        train_scores_dict = _normalize_score_results(results["train_scores"])

    for name in test_scores_dict:
        ret['test_%s' % name] = test_scores_dict[name]
        if return_train_score:
            key = 'train_%s' % name
            ret[key] = train_scores_dict[name]

    return ret


def latex_to_pandas(path):
    tab = Table.read(path).to_pandas()
    # todo: in the latex version there might be necessary to remove some parts like \toprule
    return tab
