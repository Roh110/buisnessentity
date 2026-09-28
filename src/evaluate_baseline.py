from collections import defaultdict

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
)
from src.data_loader import load_entities, iter_ground_truth
from src.fast_matcher import match_source


def load_truth(path):
    truth = {}

    for row in iter_ground_truth(path):
        truth[row["source1_entity_id"]] = set(
            row["matched_entity_ids"]
        )

    return truth


def evaluate(predictions, truth):
    total_tp = 0
    total_fp = 0
    total_fn = 0

    f05_scores = []

    for source1_id, true_matches in truth.items():
        predicted = set(predictions.get(source1_id, []))

        tp = len(predicted & true_matches)
        fp = len(predicted - true_matches)
        fn = len(true_matches - predicted)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0.0
        )

        if precision == 0 and recall == 0:
            f05 = 0.0
        else:
            beta2 = 0.5 ** 2
            f05 = (
                (1 + beta2)
                * precision
                * recall
                / (beta2 * precision + recall)
            )

        f05_scores.append(f05)

    macro_f05 = (
        sum(f05_scores) / len(f05_scores)
        if f05_scores
        else 0.0
    )

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp
        else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn
        else 0.0
    )

    return precision, recall, macro_f05


def main():
    print("Loading training Source 1...")
    source1 = load_entities(TRAIN_SOURCE1)

    print("Loading training Source 2...")
    source2 = load_entities(TRAIN_SOURCE2)

    print("Loading training Source 3...")
    source3 = load_entities(TRAIN_SOURCE3)

    print("Loading ground truth...")
    truth = load_truth(TRAIN_GROUND_TRUTH)

    print("\nMatching Source 1 -> Source 2...")
    matches2 = match_source(source1, source2)

    print("\nMatching Source 1 -> Source 3...")
    matches3 = match_source(source1, source3)

    predictions = {}

    for source1_id in source1["entity_id"].astype(str):
        combined = set()

        combined.update(matches2.get(source1_id, []))
        combined.update(matches3.get(source1_id, []))

        predictions[source1_id] = combined

    precision, recall, macro_f05 = evaluate(
        predictions,
        truth,
    )

    print("\n" + "=" * 60)
    print("TRAINING BASELINE EVALUATION")
    print("=" * 60)
    print(f"Precision : {precision:.6f}")
    print(f"Recall    : {recall:.6f}")
    print(f"Macro F0.5: {macro_f05:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
