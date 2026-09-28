from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
)


def calculate_metrics(y_true, y_pred):
    """
    Calculate precision, recall and F0.5.
    """

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f05 = fbeta_score(
        y_true,
        y_pred,
        beta=0.5,
        zero_division=0,
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f0.5": float(f05),
    }


def macro_f05(y_true, y_pred):
    """
    Calculate F0.5 using macro averaging.
    """

    return float(
        fbeta_score(
            y_true,
            y_pred,
            beta=0.5,
            average="macro",
            zero_division=0,
        )
    )


def evaluate_predictions(y_true, probabilities, threshold):
    """
    Convert probabilities into predictions using a threshold
    and calculate evaluation metrics.
    """

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    return calculate_metrics(
        y_true,
        y_pred,
    )


def print_metrics(metrics):
    """Print metrics in a readable format."""

    print("\n===== Evaluation =====")

    for name, value in metrics.items():
        print(
            f"{name}: {value:.4f}"
        )