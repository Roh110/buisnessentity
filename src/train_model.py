import joblib

from sklearn.ensemble import HistGradientBoostingClassifier


# Features used by the model
FEATURE_COLUMNS = [
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "address_ratio",
    "address_partial_ratio",
    "address_token_sort_ratio",
    "address_token_set_ratio",
    "name_jaccard",
    "address_jaccard",
    "name_length_similarity",
    "address_length_similarity",
    "country_exact",
]


def train_model(X, y):
    """
    Train the entity-resolution classifier.
    """

    X_train = X[FEATURE_COLUMNS]

    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )

    model.fit(X_train, y)

    return model


def predict_probabilities(model, X):
    """
    Return probability that each candidate pair is a match.
    """

    X_predict = X[FEATURE_COLUMNS]

    return model.predict_proba(X_predict)[:, 1]


def save_model(model, file_path):
    """Save trained model to disk."""
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, file_path)


def load_model(file_path):
    """Load a trained model from disk."""
    return joblib.load(file_path)