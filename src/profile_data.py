from __future__ import annotations

from collections import Counter

import pandas as pd

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
)
from src.data_loader import iter_entities


def profile_source(path, name: str) -> None:
    print("\n" + "=" * 70)
    print(f"PROFILING {name}")
    print("=" * 70)

    total_rows = 0

    empty_name = 0
    empty_address = 0
    empty_country = 0

    entity_ids = set()
    business_names = set()
    business_addresses = set()

    duplicate_entity_ids = 0
    duplicate_business_names = 0
    duplicate_addresses = 0

    country_counts = Counter()

    name_lengths = []
    address_lengths = []

    for chunk_number, df in enumerate(
        iter_entities(path, chunk_size=100_000),
        start=1,
    ):
        total_rows += len(df)

        # ----------------------------------------------------
        # Normalize only for profiling.
        # Raw dataset is NOT modified.
        # ----------------------------------------------------

        names = df["business_name"].astype("string").str.strip()
        addresses = df["business_address"].astype("string").str.strip()
        countries = df["country"].astype("string").str.strip()
        ids = df["entity_id"].astype("string").str.strip()

        # ----------------------------------------------------
        # Empty values
        # ----------------------------------------------------

        empty_name += (names == "").sum()
        empty_address += (addresses == "").sum()
        empty_country += (countries == "").sum()

        # ----------------------------------------------------
        # Unique / duplicate values
        # ----------------------------------------------------

        for value in ids:
            if value in entity_ids:
                duplicate_entity_ids += 1
            else:
                entity_ids.add(value)

        for value in names:
            if value:
                if value in business_names:
                    duplicate_business_names += 1
                else:
                    business_names.add(value)

        for value in addresses:
            if value:
                if value in business_addresses:
                    duplicate_addresses += 1
                else:
                    business_addresses.add(value)

        # ----------------------------------------------------
        # Country distribution
        # ----------------------------------------------------

        country_counts.update(
            value for value in countries if value
        )

        # ----------------------------------------------------
        # String lengths
        # ----------------------------------------------------

        name_lengths.extend(
            names.str.len().dropna().tolist()
        )

        address_lengths.extend(
            addresses.str.len().dropna().tolist()
        )

        if chunk_number % 10 == 0:
            print(
                f"Processed {total_rows:,} rows...",
                flush=True,
            )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    def percentile(values, q):
        if not values:
            return 0

        return int(pd.Series(values).quantile(q))

    print("\n----- ROW COUNTS -----")
    print(f"Total rows              : {total_rows:,}")

    print("\n----- EMPTY VALUES -----")
    print(f"Empty business_name     : {empty_name:,}")
    print(f"Empty business_address  : {empty_address:,}")
    print(f"Empty country           : {empty_country:,}")

    print("\n----- ENTITY IDs -----")
    print(f"Unique entity IDs       : {len(entity_ids):,}")
    print(f"Duplicate entity IDs    : {duplicate_entity_ids:,}")

    print("\n----- BUSINESS NAMES -----")
    print(f"Unique non-empty names  : {len(business_names):,}")
    print(f"Repeated name rows      : {duplicate_business_names:,}")

    print("\n----- ADDRESSES -----")
    print(f"Unique non-empty addresses : {len(business_addresses):,}")
    print(f"Repeated address rows      : {duplicate_addresses:,}")

    print("\n----- NAME LENGTH -----")
    print(f"Minimum                 : {min(name_lengths):,}")
    print(f"Median                  : {percentile(name_lengths, 0.50):,}")
    print(f"95th percentile         : {percentile(name_lengths, 0.95):,}")
    print(f"Maximum                 : {max(name_lengths):,}")

    print("\n----- ADDRESS LENGTH -----")
    print(f"Minimum                 : {min(address_lengths):,}")
    print(f"Median                  : {percentile(address_lengths, 0.50):,}")
    print(f"95th percentile         : {percentile(address_lengths, 0.95):,}")
    print(f"Maximum                 : {max(address_lengths):,}")

    print("\n----- TOP COUNTRIES -----")
    for country, count in country_counts.most_common(20):
        print(f"{country:20} {count:,}")


def main() -> None:
    profile_source(TRAIN_SOURCE1, "SOURCE 1")
    profile_source(TRAIN_SOURCE2, "SOURCE 2")
    profile_source(TRAIN_SOURCE3, "SOURCE 3")


if __name__ == "__main__":
    main()