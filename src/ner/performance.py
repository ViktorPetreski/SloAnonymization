import pandas as pd
from typing import List, Union
from sklearn.metrics import precision_recall_fscore_support
import warnings

def flatten(l: list):
    """Flattens list"""
    return [item for sublist in l for item in sublist]


def compute_f1_scores(y_pred: List[List[str]],
                      y_true: List[List[str]],
                      labels: List[str],
                      **kwargs) -> list:
    """Compute F1 scores.

    Computes F1 Scores

    Args:
        y_pred (List): predicted values.
        y_true (List): observed/true values.
        labels (List): all possible tags.
        kwargs: all optional arguments for precision/recall function.

    Returns:
        list: resulting F1 scores.

    """
    # check inputs.
    assert sum([len(t) < len(p) for t, p in zip(y_true, y_pred)]) == 0, "Length of predictions must not exceed length of observed values"

    # check, if some lengths of observed values exceed predicted values.
    n_exceeds = sum([len(t) > len(p) for t, p in zip(y_true, y_pred)])
    if n_exceeds > 0:
        warnings.warn(f'length of observed values exceeded lengths of predicted values in {n_exceeds} cases and were truncated. _Consider_ increasing max_len parameter for your model.')

    # truncate observed values dimensions to match predicted values,
    # this is needed if predictions have been truncated earlier in
    # the flow.
    y_true = [t[:len(p)] for t, p in zip(y_true, y_pred)]

    y_pred = flatten(y_pred)
    y_true = flatten(y_true)

    f1_scores = precision_recall_fscore_support(y_true = y_true,
                                                y_pred = y_pred,
                                                labels = labels,
                                                **kwargs)

    return f1_scores


def evaluate_performance(y_pred:List[List[Union[str, int]]],
                         y_true:List[List[Union[str, int]]],
                         labels:List[str],
                         **kwargs) -> pd.DataFrame:
    """Evaluate Performance

    Evaluates the performance of the model on an arbitrary
    data set.

    Args:
        y_pred (list): Predicted labels
        y_true (list): Ground truth
        labels (list): List of class labels
        kwargs: arbitrary keyword arguments for predict. For
            instance 'batch_size' and 'num_workers'.

    Returns:
        DataFrame with performance numbers, F1-scores.
    """


    f1 = compute_f1_scores(y_pred = y_pred,
                           y_true = y_true,
                           labels = labels,
                           average = None)

    # create DataFrame with performance scores (=F1)
    df = list(zip(labels, f1[0], f1[1], f1[2]))
    df = pd.DataFrame(df, columns = ['Level', "Precision", "Recall", 'F1-Score'])

    # compute MICRO-averaged F1-scores and add to table.
    f1_micro = compute_f1_scores(y_pred = y_pred,
                                 y_true = y_true,
                                 labels = labels,
                                 average = 'micro')
    f1_micro = pd.DataFrame({'Level' : ['AVG_MICRO'], "Precision": [f1_micro[0]], "Recall": [f1_micro[1]], 'F1-Score': [f1_micro[2]]})
    df = df.append(f1_micro)

    # compute MACRO-averaged F1-scores and add to table.
    f1_macro = compute_f1_scores(y_pred = y_pred,
                                 y_true = y_true,
                                 labels = labels,
                                 average = 'macro')
    f1_macro = pd.DataFrame({'Level' : ['AVG_MACRO'], "Precision": [f1_macro[0]], "Recall": [f1_macro[1]], 'F1-Score': [f1_macro[2]]})
    df = df.append(f1_macro)
    df = df.round({"Precision": 2, "Recall": 2, "F1-Score": 3})
    return df