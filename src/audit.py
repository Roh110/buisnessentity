from __future__ import annotations

from src.config import (
    TRAIN_GROUND_TRUTH,
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
)
from src.data_loader import (
    iter_ground_truth,
    load_entities,
)


def inspect_source(df, name: str) -> None:
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nEmpty strings:")
    for col in ["business_name", "business_address", "country"]:
        if col in df.columns:
            empty_count = (df[col].str.strip() == "").sum()
            print(f"{col}: {empty_count:,}")

    print("\nCountries:")
    print(df["country"].value_counts(dropna=False).head(20))

    print("\nDuplicate entity IDs:")
    print(df["entity_id"].duplicated().sum())

    print("\nSample:")
    print(df.head(5).to_string(index=False))


def inspect_ground_truth(path) -> None:
    print("\n" + "=" * 60)
    print("GROUND TRUTH")
    print("=" * 60)

    match_counts = []
    total_rows = 0

    for row in iter_ground_truth(path):
        total_rows += 1

        # iter_ground_truth() returns matched_entity_ids as a list.
        matches = row.get("matched_entity_ids")
        if matches is None:
            matches = []

        match_counts.append(len(matches))

    print("Rows:", f"{total_rows:,}")

    print("\nMatch-count distribution:")

    distribution = {}

    for count in match_counts:
        distribution[count] = distribution.get(count, 0) + 1

    for count in sorted(distribution):
        print(f"{count:>3} matches : {distribution[count]:,}")

    print(
        "\nSingletons / no-match:",
        f"{distribution.get(0, 0):,}",
    )

    print(
        "One match:",
        f"{distribution.get(1, 0):,}",
    )

    print(
        "Multiple matches:",
        f"{sum(v for k, v in distribution.items() if k > 1):,}",
    )

    print(
        "\nMaximum matches for one Source 1 entity:",
        max(match_counts),
    )


def main() -> None:
    print("Loading Source 1...")
    source1 = load_entities(TRAIN_SOURCE1)

    print("Loading Source 2...")
    source2 = load_entities(TRAIN_SOURCE2)

    print("Loading Source 3...")
    source3 = load_entities(TRAIN_SOURCE3)

    inspect_source(source1, "SOURCE 1")
    inspect_source(source2, "SOURCE 2")
    inspect_source(source3, "SOURCE 3")

    inspect_ground_truth(TRAIN_GROUND_TRUTH)


if __name__ == "__main__":
    main()