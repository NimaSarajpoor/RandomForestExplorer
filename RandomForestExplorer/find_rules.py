import numpy as np
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

from .find_paths import get_decision_paths


def _get_paths_relaxer(decisions_paths):
    """
    Given a list of decision_paths, it relax them by merging the
    decision_maker_values. For instance, the triples (0, 1.4, 1) and (0, 1.6, 1)
    show that the feature at index 0 is less than 1.4 (former) and less than 1.6
    (latter). We can make them similar by using the median of 1.4 and 1.6 of
    instead of 1.4 or 1.6. We only consider trees that their prediction is the
    same as y_pred, the prediction of random forest.

    Parameters
    ----------
    decisions_paths : List
        A nested list of lists, where decision_paths[i] is a list containing the
        decisions path of x obtained from the i-th decision tree. A decision
        path is a set of triples (idx, decision_maker_value, indicator), where
        `idx` is the index of a feature, decision_maker_value is the value that
        is used to split the idx-th feature, and indicator is a binary value,
        where 1 means the value of such feature in observation `x` is below
        `decision_maker_value`. 0 otherwise. For example, for random forest with
        two decision trees, we may have:
        decisions_paths=[[(1, 3.1, 0)], [(1, 2.5, 1),(0, 7.3, 0)]].

    Returns
    -------
    mapper : dict
        A dictionary with key=(feature_idx, indicator) and value as the
        decision_maker_value
    """
    lst = []
    for item in decisions_paths:
        for triple in item:  # triple: tuple with three elements
            lst.append((triple[0], triple[2]))

    relaxing_mapper = {item: [] for item in set(lst)}
    for item in decisions_paths:
        for triple in item:
            relaxing_mapper[(triple[0], triple[2])].append(triple[1])

    for key in relaxing_mapper:  # using one bin in digitizing
        relaxing_mapper[key] = np.median(relaxing_mapper[key])

    return relaxing_mapper


def _relax_paths(decisions_paths):
    """
    Parameters
    ----------
    decisions_paths : List
        A nested list of lists, where decision_paths[i] is a list containing the
        decisions path of x obtained from a decision tree. A decision path
        is a set of triples (idx, decision_maker_value, indicator), where `idx`
        is the index of a feature, decision_maker_value is the value that is used
        to split the idx-th feature, and indicator is a binary value, where 1
        means the value of such feature in observation `x` is below
        `decision_maker_value`. 0 otherwise. For example, for random forest with
        two decision trees, we may have:
        decisions_paths=[[(1, 3.1, 0)], [(1, 2.5, 1),(0, 7.3, 0)]]

    Returns
    -------
    out : List
        A nest list, where each list contains triples that are relaxed.
    """
    relaxing_mapper = _get_paths_relaxer(decisions_paths)

    out = []
    for item in decisions_paths:
        lst = []
        for triple in item:
            key = (triple[0], triple[2])
            lst.append((triple[0], relaxing_mapper[key], triple[2]))

        out.append(lst)

    return out


def _find_rules(rf, x, y_true, min_support=0.2):
    """
    Given a fitted random forest classifier `rf`, return a set of rules extracted
    from decision paths uisng frequent mining.

    Parameters
    ----------
    rf : a fitted, random forest binary classifier
        A random forest classiier, instanciated from sklearn random forest class,
        and fitted.

    x : numpy.ndarray
        A 1D array of real values correponding to one observation.

    y_true : int, binary
        The groundtruth class ID of observation `x`

    min_support : float, default 0.2
        The minimum support an item should have for being considered as frequent

    Returns
    -------
    freq_transactions : pandas.DataFrame
        A dataframe with two columsn. Each row corresponds to a particular
        freq transction. First column is the support of freq item and the second
        column is the freq item itself.

    is_pred_correct : bool
        True if `rf` can predict `x` correctly. False otherwise.
    """
    decisions_paths, decisions_values, y_pred = get_decision_paths(rf, x)

    if y_pred == y_true:
        is_pred_correct = True
    else:
        is_pred_correct = False

    # keep only the paths whose corresponding tree predicts y_pred
    decisions_paths_pruned = []
    for i, item in enumerate(decisions_paths):
        if decisions_values[i] == y_pred:
            decisions_paths_pruned.append(item)

    transactions = _relax_paths(decisions_paths_pruned)

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    freq_transactions = apriori(df, min_support=min_support, use_colnames=True)

    return freq_transactions, is_pred_correct


def find_rules(rf, x, y_true, min_support=0.2):
    """
    Given a fitted random forest classifier `rf`, return a set of rules extracted
    from decision paths uisng frequent mining. This is a convenient wrapper around
    _find_rules.

    Parameters
    ----------
    rf : a fitted, random forest binary classifier
        A random forest classiier, instanciated from sklearn random forest class,
        and fitted.

    x : numpy.ndarray
        A 1D array of real values correponding to one observation.

    y_true : int, binary
        The groundtruth class ID of observation `x`

    min_support : float, default 0.2
        The minimum support an item should have for being considered as frequent

    Returns
    -------
    freq_transactions : pandas.DataFrame
        A dataframe with two columsn. Each row corresponds to a particular
        freq transction. First column is the support of freq item and the second
        column is the freq item itself.

    is_pred_correct : bool
        True if `rf` can predict `x` correctly. False otherwise.
    """
    freq_transactions, is_pred_correct = _find_rules(rf, x, y_true, min_support)

    return freq_transactions, is_pred_correct


def find_rules_on_samples(rf, X, y_true, min_support=0.2):
    """
    Given a fitted random forest classifier `rf`, return a set of rules extracted
    from decision paths uisng frequent mining. This is a convenient wrapper around
    _find_rules.

    Parameters
    ----------
    rf : a fitted, random forest binary classifier
        A random forest classiier, instanciated from sklearn random forest class,
        and fitted.

    X:  : numpy.ndarray
        A 2D array of real values, where each row corresponds to one observation.
        Users must ensure that all observations in X are in the same class.

    y_true : int, binary
        The groundtruth class ID of observation `x`

    min_support : float, default 0.2
        The minimum support an item should have for being considered as frequent

    Returns
    -------
    freq_transactions : pandas.DataFrame
        A dataframe with two columsn. Each row corresponds to a particular
        freq transction. First column is the support of freq item and the second
        column is the freq item itself.
    """
    if X.ndim != 2:
        raise ValueError(f"The array `X` must be 2D. Got {X.ndim} for size")

    all_decision_paths = []
    for x in X:
        decisions_paths, y_pred_of_trees, y_pred_rf = get_decision_paths(rf, x)
        if y_pred_ref == y_true:
            y_pred_of_trees = np.array(y_pred_of_trees)
            IDX = np.flatnonzero(y_pred_of_trees == y_true)
            for idx in IDX:
                all_decision_paths.append(decisions_paths[idx])

    transactions = _relax_paths(all_decision_paths)
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    freq_transactions = apriori(df, min_support=min_support, use_colnames=True)

    return freq_transactions
