from __future__ import annotations

from pathlib import Path
import pandas as pd


BASE = Path(__file__).resolve().parents[1]

TRAIN_GT = BASE / "data" / "raw" / "extracted" / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv"

MATCHING = BASE / "output" / "matching_results.tsv"
CANDIDATES = BASE / "output" / "candidate_pairs.tsv"


def parse_ids(value: str) -> set[str]:
    if not value:
        return set()

    return {
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    }


def main():
    print("=" * 70)
    print("CURRENT SOLUTION DIAGNOSTICS")
    print("=" * 70)

    print("\nLoading training ground truth...")
    gt = pd.read_csv(
        TRAIN_GT,
        sep="\t",
        dtype="string",
        keep_default_na=False,
    )

    print("Ground truth rows:", len(gt))

    print("\nLoading candidate output...")
    candidates = pd.read_csv(
        CANDIDATES,
        sep="\t",
        dtype="string",
        keep_default_na=False,
    )

    print("Candidate rows:", len(candidates))

    print("\nLoading matching output...")
    matching = pd.read_csv(
        MATCHING,
        sep="\t",
        dtype="string",
        keep_default_na=False,
    )

    print("Matching rows:", len(matching))

    match_counts = matching["matched_entity_ids"].map(
        lambda x: len(parse_ids(x))
    )

    candidate_counts = candidates["candidate_entity_ids"].map(
        lambda x: len(parse_ids(x))
    )

    print()
    print("MATCH DISTRIBUTION")
    print("-" * 70)

    print("Source 1 rows:", len(matching))
    print("Non-empty:", int((match_counts > 0).sum()))
    print("Empty:", int((match_counts == 0).sum()))
    print("Total predicted links:", int(match_counts.sum()))
    print("Average links/S1:", float(match_counts.mean()))
    print("Maximum links/S1:", int(match_counts.max()))

    print()
    print("CANDIDATE DISTRIBUTION")
    print("-" * 70)

    print("Total candidates:", int(candidate_counts.sum()))
    print("Average candidates/S1:", float(candidate_counts.mean()))
    print("Maximum candidates/S1:", int(candidate_counts.max()))

    print()
    print("PREDICTED LINK COUNTS")
    print("-" * 70)

    for n in [0, 1, 2, 3, 4, 5, 10, 20, 50]:
        print(
            f"<= {n:2d}:",
            int((match_counts <= n).sum())
        )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()