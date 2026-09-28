import numpy as np

from sklearn.metrics import fbeta_score


def find_best_threshold(
    y_true,
    probabilities,
    min_threshold=0.30,
    max_threshold=0.99,
    step=0.01,
):
    """
    Search for the threshold that gives the highest F0.5.
    """

    best_threshold = min_threshold
    best_score = -1.0

    thresholds = np.arange(
        min_threshold,
        max_threshold + step,
        step,
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = fbeta_score(
            y_true,
            predictions,
            beta=0.5,
            zero_division=0,
        )

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return {
        "threshold": best_threshold,
        "f0.5": float(best_score),
    }


def print_threshold_result(result):
    """Display the selected threshold."""

    print("\n===== Threshold Optimization =====")

    print(
        f"Best threshold: "
        f"{result['threshold']:.2f}"
    )

    print(
        f"Best F0.5: "
        f"{result['f0.5']:.4f}"
    )