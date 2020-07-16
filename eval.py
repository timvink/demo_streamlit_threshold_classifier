import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score


def get_metrics_df(y_train, yhat_prob_train, y_test, yhat_prob_test, threshold=0.5):
    """
    Returns a dataframe with relevant model performance metrics

    Args:
        y_train (np.array): actuals train, integers, 0 or 1
        yhat_prob_train (np.array): predicted probabilities train, float 0-1
        y_test (np.array): actuals test, integers, 0 or 1
        yhat_prob_test (np.array): predicted probabilities test, float 0-1
        threshold (float, optional): Threshold for probabilities. Defaults to 0.5.

    Returns:
        pd.DataFrame: metrics DF
    """

    yhat_train = np.where(yhat_prob_train >= threshold, 1, 0)
    yhat_test = np.where(yhat_prob_test >= threshold, 1, 0)

    return pd.DataFrame(
        {
            "type": ["auc", "recall", "precision", "accuracy"],
            "cv-train": [
                roc_auc_score(y_train, yhat_prob_train),
                recall_score(y_train, yhat_train),
                precision_score(y_train, yhat_train),
                accuracy_score(y_train, yhat_train),
            ],
            "test": [
                roc_auc_score(y_test, yhat_prob_test),
                recall_score(y_test, yhat_test),
                precision_score(y_test, yhat_test),
                accuracy_score(y_test, yhat_test),
            ],
        }
    )
