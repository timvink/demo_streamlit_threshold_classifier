import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split


def load_data(n_samples=30_000):
    """
    Loads sample data

    Args:
        n_samples (int, optional): Number of data samples. Defaults to 30_000.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """

    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=2,
        n_classes=2,
        weights=[0.9, 0.1],
        random_state=42,
    )
    X = pd.DataFrame(X)

    return train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


def train_model(X, y):
    """Trains cross-validated tuned model

    Args:
        X (pd.DataFrame): training data
        y (np.array): target data

    Returns:
        sklearn.ensemble.RandomForestClassifier: The best cross-validated model (refit on X)
    """

    pipe = Pipeline(
        [
            (
                "rf",
                RandomForestClassifier(
                    max_depth=5,
                    n_estimators=10,
                    random_state=42,
                    class_weight="balanced",
                ),
            )
        ]
    )

    # Very small grid for demo purpose only
    grid = {"rf__max_depth": [4, 7]}

    mod = GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        scoring={"auc": make_scorer(roc_auc_score)},
        refit="auc",
        cv=5,
    )

    mod.fit(X, y)

    return mod.best_estimator_


def get_predictions():
    """
    Get predictions from model

    Returns:
        tuple: y_train, yhat_prob_train, y_test, yhat_prob_test
    """

    X_train, X_test, y_train, y_test = load_data()
    clf = train_model(X_train, y_train)

    yhat_prob_train = clf.predict_proba(X_train)[:, 1]
    yhat_prob_test = clf.predict_proba(X_test)[:, 1]

    return y_train, yhat_prob_train, y_test, yhat_prob_test
