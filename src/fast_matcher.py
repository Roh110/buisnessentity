from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.config import (
    OUTPUT_DIR,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
)
from src.data_loader import load_entities
from src.normalization import (
    normalize_address,
    normalize_business_name,
    normalize_country,
)


def build_name_index(
    df: pd.DataFrame,
) -> dict[tuple[str, str], list[str]]:
    """Build country + normalized business name -> entity IDs."""

    index: dict[tuple[str, str], list[str]] = defaultdict(list)

    entity_ids = df["entity_id"].astype(str)
    names = df["business_name"].astype(str)
    countries = df["country"].astype(str)

    for entity_id, name, country in zip(
        entity_ids,
        names,
        countries,
    ):
        normalized_name = normalize_business_name(name)
        normalized_country = normalize_country(country)

        if normalized_name:
            index[
                (normalized_country, normalized_name)
            ].append(entity_id)

    return dict(index)


def build_address_index(
    df: pd.DataFrame,
) -> dict[tuple[str, str], list[str]]:
    """Build country + normalized address -> entity IDs."""

    index: dict[tuple[str, str], list[str]] = defaultdict(list)

    entity_ids = df["entity_id"].astype(str)
    addresses = df["business_address"].astype(str)
    countries = df["country"].astype(str)

    for entity_id, address, country in zip(
        entity_ids,
        addresses,
        countries,
    ):
        normalized_address = normalize_address(address)
        normalized_country = normalize_country(country)

        if normalized_address:
            index[
                (normalized_country, normalized_address)
            ].append(entity_id)

    return dict(index)


def match_source(
    source1: pd.DataFrame,
    target: pd.DataFrame,
) -> dict[str, list[str]]:
    """
    Fast exact matching between Source 1 and one target source.

    Blocking rule:
        Source 1 country == target country

    Matching signals:
        1. Exact normalized business name
        2. Exact normalized business address

    Both signals are allowed to contribute candidates.
    """

    name_index = build_name_index(target)
    address_index = build_address_index(target)

    results: dict[str, list[str]] = {}

    source1_ids = source1["entity_id"].astype(str)
    source1_names = source1["business_name"].astype(str)
    source1_addresses = source1["business_address"].astype(str)
    source1_countries = source1["country"].astype(str)

    total = len(source1)

    for counter, (
        source1_id,
        source1_name,
        source1_address,
        source1_country,
    ) in enumerate(
        zip(
            source1_ids,
            source1_names,
            source1_addresses,
            source1_countries,
        ),
        start=1,
    ):
        country = normalize_country(source1_country)
        name = normalize_business_name(source1_name)
        address = normalize_address(source1_address)

        candidates: set[str] = set()

        # ---------------------------------------------------------
        # 1. Exact normalized business name + same country
        # ---------------------------------------------------------
        if name:
            candidates.update(
                name_index.get(
                    (country, name),
                    [],
                )
            )

        # ---------------------------------------------------------
        # 2. Exact normalized address + same country
        # ---------------------------------------------------------
        if address:
            candidates.update(
                address_index.get(
                    (country, address),
                    [],
                )
            )

        # Remove accidental self-match just in case.
        candidates.discard(source1_id)

        results[source1_id] = sorted(candidates)

        if counter % 100_000 == 0:
            print(
                f"Processed {counter:,} / {total:,} "
                "Source 1 entities"
            )

    return results


def write_results(
    results: dict[str, list[str]],
    path: Path,
) -> None:
    """Write matching results in official TSV format."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [
        {
            "source1_entity_id": source1_id,
            "matched_entity_ids": ",".join(matches),
        }
        for source1_id, matches in results.items()
    ]

    output_df = pd.DataFrame(
        rows,
        columns=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )

    output_df.to_csv(
        path,
        sep="\t",
        index=False,
    )


def main() -> None:
    print("Loading test Source 1...")
    source1 = load_entities(TEST_SOURCE1)

    print("Loading test Source 2...")
    source2 = load_entities(TEST_SOURCE2)

    print("Loading test Source 3...")
    source3 = load_entities(TEST_SOURCE3)

    print("\nMatching Source 1 -> Source 2...")
    matches2 = match_source(
        source1,
        source2,
    )

    print("\nMatching Source 1 -> Source 3...")
    matches3 = match_source(
        source1,
        source3,
    )

    final_results: dict[str, list[str]] = {}

    for source1_id in source1["entity_id"].astype(str):
        combined: set[str] = set()

        combined.update(
            matches2.get(source1_id, [])
        )

        combined.update(
            matches3.get(source1_id, [])
        )

        final_results[source1_id] = sorted(combined)

    output_path = OUTPUT_DIR / "matching_results.tsv"

    write_results(
        final_results,
        output_path,
    )

    matched_entities = sum(
        bool(matches)
        for matches in final_results.values()
    )

    total_links = sum(
        len(matches)
        for matches in final_results.values()
    )

    print("\n" + "=" * 60)
    print("MATCHING COMPLETE")
    print("=" * 60)

    print(
        f"Source 1 entities : {len(final_results):,}"
    )

    print(
        f"Entities matched  : {matched_entities:,}"
    )

    print(
        f"Total match links : {total_links:,}"
    )

    print(
        f"Output            : {output_path}"
    )


if __name__ == "__main__":
    main()