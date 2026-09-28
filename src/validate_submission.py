from pathlib import Path
import pandas as pd

from src.config import (
    OUTPUT_DIR,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
)

CANDIDATE_FILE = OUTPUT_DIR / "candidate_pairs.tsv"
MATCHING_FILE = OUTPUT_DIR / "matching_results.tsv"


def load_tsv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
    )

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in {path.name}: {missing}"
        )

    return df


def load_ids(path: Path) -> set[str]:
    df = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        usecols=["entity_id"],
        keep_default_na=False,
    )
    return set(df["entity_id"].astype(str))


def validate_candidate_pairs() -> pd.DataFrame:
    df = load_tsv(
        CANDIDATE_FILE,
        ["source1_entity_id", "candidate_entity_ids"],
    )

    test_ids = load_ids(TEST_SOURCE1)

    actual_ids = set(df["source1_entity_id"])
    missing = test_ids - actual_ids
    extra = actual_ids - test_ids

    if missing:
        raise ValueError(
            f"Candidate pairs missing {len(missing):,} Source 1 IDs"
        )

    if extra:
        raise ValueError(
            f"Candidate pairs contain {len(extra):,} invalid Source 1 IDs"
        )

    if df["source1_entity_id"].duplicated().any():
        raise ValueError(
            "Duplicate source1_entity_id values in candidate_pairs.tsv"
        )

    valid_target_ids = (
        load_ids(TEST_SOURCE2)
        | load_ids(TEST_SOURCE3)
    )

    bad_targets = 0

    for raw in df["candidate_entity_ids"]:
        if not raw:
            continue

        ids = [x.strip() for x in raw.split(",") if x.strip()]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Duplicate candidate entity IDs found"
            )

        for entity_id in ids:
            if entity_id not in valid_target_ids:
                bad_targets += 1

    if bad_targets:
        raise ValueError(
            f"Found {bad_targets:,} invalid candidate target IDs"
        )

    return df


def validate_matching_results(
    candidate_df: pd.DataFrame,
) -> pd.DataFrame:
    df = load_tsv(
        MATCHING_FILE,
        ["source1_entity_id", "matched_entity_ids"],
    )

    if df["source1_entity_id"].duplicated().any():
        raise ValueError(
            "Duplicate source1_entity_id values in matching_results.tsv"
        )

    candidate_map = dict(
        zip(
            candidate_df["source1_entity_id"],
            candidate_df["candidate_entity_ids"],
        )
    )

    test_ids = load_ids(TEST_SOURCE1)
    actual_ids = set(df["source1_entity_id"])

    missing = test_ids - actual_ids
    extra = actual_ids - test_ids

    if missing:
        raise ValueError(
            f"Matching results missing {len(missing):,} Source 1 IDs"
        )

    if extra:
        raise ValueError(
            f"Matching results contain {len(extra):,} invalid Source 1 IDs"
        )

    for _, row in df.iterrows():
        source1_id = row["source1_entity_id"]
        matches = [
            x.strip()
            for x in row["matched_entity_ids"].split(",")
            if x.strip()
        ]

        if len(matches) != len(set(matches)):
            raise ValueError(
                f"Duplicate matches for {source1_id}"
            )

        candidates = [
            x.strip()
            for x in candidate_map.get(source1_id, "").split(",")
            if x.strip()
        ]

        candidate_set = set(candidates)

        outside_candidates = [
            x for x in matches
            if x not in candidate_set
        ]

        if outside_candidates:
            raise ValueError(
                f"Matches outside candidate set for {source1_id}"
            )

    return df


def main() -> None:
    print("=" * 60)
    print("VALIDATING SUBMISSION")
    print("=" * 60)

    candidate_df = validate_candidate_pairs()

    print(
        f"Candidate pairs : {len(candidate_df):,} Source 1 rows"
    )

    matching_df = validate_matching_results(candidate_df)

    print(
        f"Matching results: {len(matching_df):,} Source 1 rows"
    )

    empty_matches = (
        matching_df["matched_entity_ids"] == ""
    ).sum()

    total_links = sum(
        len(
            [
                x
                for x in value.split(",")
                if x.strip()
            ]
        )
        for value in matching_df["matched_entity_ids"]
    )

    print(f"Empty matches   : {empty_matches:,}")
    print(f"Total match links: {total_links:,}")

    print()
    print("=" * 60)
    print("VALIDATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
