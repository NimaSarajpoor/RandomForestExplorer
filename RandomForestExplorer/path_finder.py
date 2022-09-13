import numpy as np
from sklearn.utils.validation import check_is_fitted

#import pandas as pd
#from mlxtend.preprocessing import TransactionEncoder
#from mlxtend.frequent_patterns import apriori

def _get_decisiion_paths(rf, x):
    """
    Given a fitted random forest classifier -of sklearn, returns the decision paths of
    tree.

    Parameters
    ----------
    rf : a fitted, random forest binary classifier
        A random forest classiier, instanciated from sklearn random forest class,
        and fitted.

    x : numpy.ndarray
        A 1D array of real values correponding to one observation.

    Returns
    -------
    decisions_paths : List
        A nested list of lists, where decision_paths[i] is a list containing the
        decisions path of x obtained from the i-th decision tree. A decision path
        is a set of triples (idx, decision_maker_value, indicator), where `idx`
        is the index of a feature, decision_maker_value is the value that is used
        to split the idx-th feature, and indicator is a binary value, where 1 means
        the value of such feature in observation `x` is below `decision_maker_value`.
        0 otherwise. For example, for random forest with two decision trees, we
        may have: decisions_paths=[[(1, 3.1, 0)], [(1, 2.5, 1),(0, 7.3, 0)]]

    decisions_values : List
        A list with length equal to the number of trees in rf. decisions_values[i]
        is the target value predicted by the i-th decision tree of random forest.

    y_pred : int, binary
        The predicted class of x
    """
    try:
        if rf.__class__.__name__ != 'RandomForestClassifier':
            raise AttributeError("The object is not random forest classifier")
        check_is_fitted(rf) # check rf is fitted
    except:
        raise ValueError("The input `rf` does not have attribute __class__")

    if x.ndim != 1:
        raise ValueError("The input `x` is not 1D array")

    x_2D = np.atleast_2d(x)
    sample_id = 0 # because `x` is the first row of x_2D

    y_pred = rf.predict(x_2D)
    y_pred = y_pred[0]

    decisions_values = []
    decisions_paths = []
    for est_idx, est in enumerate(rf.estimators_):
        y_pred_tree = est.predict(x_2D)
        decisions_values.append(y_pred_tree[0])

        DTree_ = est.tree_
        features_id = DTree_.feature
        thresholds_val = DTree_.threshold

        node_indicator = DTree_.decision_path(x_2D)
        leaf_id = DTree_.apply(x_2D)

        node_index = node_indicator.indices[
            node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
        ]

        transactions = []
        for node_id in node_index:
            if leaf_id[sample_id] == node_id:
                continue # continue to the next node if it is a leaf node

            feature_idx = feature=features_id[node_id]
            threshold = thresholds_val[node_id]

            if x_2D[sample_id, feature_idx] < threshold:
                is_below_threshold = 1
            else:
                is_below_threshold = 0

            transactions.append((feature, threshold, is_below_threshold))

        decisions_paths.append(transactions)

    return decisions_paths,  decisions_values, y_pred
