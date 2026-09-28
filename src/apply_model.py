from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd

from src.config import (
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
    OUTPUT_DIR,
)
from src.normalization import (
    normalize_business_name,
    normalize_address,
    normalize_country,
)
from src.ml_matcher import (
    FEATURE_NAMES,
    make_features,
)


# ============================================================
# SETTINGS
# ============================================================

CANDIDATE_FILE = (
    OUTPUT_DIR
    / "candidate_pairs.tsv"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "matching_results.tsv"
)

MODEL_FILE = (
    OUTPUT_DIR
    / "entity_matcher.joblib"
)

TARGET_CHUNK_SIZE = 500_000

PREDICT_BATCH_SIZE = 100_000


# ============================================================
# LOAD SOURCE 1
# ============================================================

def load_source1() -> pd.DataFrame:

    print(
        "Loading Source 1..."
    )

    source1 = pd.read_csv(
        TEST_SOURCE1,
        sep="\t",
        dtype="string",
        keep_default_na=False,
    )

    source1["name_norm"] = (
        source1["business_name"]
        .astype(str)
        .map(normalize_business_name)
    )

    source1["address_norm"] = (
        source1["business_address"]
        .astype(str)
        .map(normalize_address)
    )

    source1["country_norm"] = (
        source1["country"]
        .astype(str)
        .map(normalize_country)
    )

    source1 = source1.set_index(
        "entity_id",
        drop=False,
    )

    print(
        f"Source 1 rows: "
        f"{len(source1):,}"
    )

    return source1


# ============================================================
# LOAD TARGETS
# ============================================================

def load_targets() -> pd.DataFrame:

    print()
    print(
        "Loading Source 2 and Source 3..."
    )

    pieces = []

    for path in [
        TEST_SOURCE2,
        TEST_SOURCE3,
    ]:

        print(
            f"Reading: {path}"
        )

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype="string",
            keep_default_na=False,
            chunksize=TARGET_CHUNK_SIZE,
        ):

            chunk["name_norm"] = (
                chunk["business_name"]
                .astype(str)
                .map(
                    normalize_business_name
                )
            )

            chunk["address_norm"] = (
                chunk["business_address"]
                .astype(str)
                .map(
                    normalize_address
                )
            )

            chunk["country_norm"] = (
                chunk["country"]
                .astype(str)
                .map(
                    normalize_country
                )
            )

            pieces.append(
                chunk[
                    [
                        "entity_id",
                        "name_norm",
                        "address_norm",
                        "country_norm",
                    ]
                ]
            )

    targets = pd.concat(
        pieces,
        ignore_index=True,
    )

    targets = targets.set_index(
        "entity_id",
        drop=False,
    )

    print(
        f"Target entities: "
        f"{len(targets):,}"
    )

    return targets


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print(
        "BUSINESS ENTITY RESOLUTION — "
        "BATCHED ML INFERENCE"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_FILE}"
        )

    artifact = joblib.load(
        MODEL_FILE
    )

    model = artifact["model"]

    trained_threshold = float(
        artifact["threshold"]
    )

    # --------------------------------------------------------
    # Optional threshold override
    # --------------------------------------------------------

    threshold_override = os.environ.get(
        "MATCH_THRESHOLD"
    )

    if threshold_override:

        threshold = float(
            threshold_override
        )

        print(
            f"Threshold override: "
            f"{threshold:.4f}"
        )

    else:

        threshold = trained_threshold

        print(
            f"Model threshold: "
            f"{threshold:.4f}"
        )

    print(
        f"Model features: "
        f"{len(FEATURE_NAMES)}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    source1 = load_source1()

    targets = load_targets()

    # --------------------------------------------------------
    # Candidate file
    # --------------------------------------------------------

    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(
            f"Candidate file not found:\n"
            f"{CANDIDATE_FILE}"
        )

    print()
    print(
        "Loading candidate pairs..."
    )

    candidates = pd.read_csv(
        CANDIDATE_FILE,
        sep="\t",
        dtype="string",
        keep_default_na=False,
    )

    print(
        f"Candidate rows: "
        f"{len(candidates):,}"
    )

    # --------------------------------------------------------
    # Prediction storage
    # --------------------------------------------------------

    predictions: Dict[
        str,
        List[str],
    ] = defaultdict(list)

    feature_rows = []

    metadata = []

    total_candidates = 0
    missing_targets = 0
    kept_matches = 0

    # --------------------------------------------------------
    # Batch flush
    # --------------------------------------------------------

    def flush_batch():

        nonlocal feature_rows
        nonlocal metadata
        nonlocal kept_matches

        if not feature_rows:
            return

        X = pd.DataFrame(
            feature_rows,
            columns=FEATURE_NAMES,
        )

        probabilities = (
            model.predict_proba(X)[:, 1]
        )

        for (
            (source1_id, candidate_id),
            probability,
        ) in zip(
            metadata,
            probabilities,
        ):

            if (
                float(probability)
                >= threshold
            ):

                predictions[
                    source1_id
                ].append(
                    candidate_id
                )

                kept_matches += 1

        feature_rows = []
        metadata = []

    # --------------------------------------------------------
    # Score candidates
    # --------------------------------------------------------

    print()
    print(
        "Scoring candidate pairs in batches..."
    )

    for row in candidates.itertuples(
        index=False
    ):

        source1_id = str(
            row.source1_entity_id
        )

        raw_candidates = str(
            row.candidate_entity_ids
        ).strip()

        if not raw_candidates:
            continue

        if source1_id not in source1.index:
            continue

        source = source1.loc[
            source1_id
        ]

        name_a = str(
            source["name_norm"]
        )

        address_a = str(
            source["address_norm"]
        )

        country_a = str(
            source["country_norm"]
        )

        candidate_ids = [
            x.strip()
            for x in raw_candidates.split(",")
            if x.strip()
        ]

        for candidate_id in candidate_ids:

            total_candidates += 1

            if candidate_id not in targets.index:

                missing_targets += 1

                continue

            target = targets.loc[
                candidate_id
            ]

            feature_rows.append(
                make_features(
                    name_a,
                    address_a,
                    country_a,
                    str(
                        target["name_norm"]
                    ),
                    str(
                        target["address_norm"]
                    ),
                    str(
                        target["country_norm"]
                    ),
                )
            )

            metadata.append(
                (
                    source1_id,
                    candidate_id,
                )
            )

            if (
                len(feature_rows)
                >= PREDICT_BATCH_SIZE
            ):

                flush_batch()

        if (
            total_candidates
            % 1_000_000
            < len(candidate_ids)
        ):

            print(
                f"Candidates processed: "
                f"{total_candidates:,} | "
                f"ML matches kept: "
                f"{kept_matches:,}"
            )

    # --------------------------------------------------------
    # Remaining batch
    # --------------------------------------------------------

    flush_batch()

    print()
    print(
        f"Candidate pairs scored: "
        f"{total_candidates:,}"
    )

    print(
        f"ML matches kept: "
        f"{kept_matches:,}"
    )

    print(
        f"Missing target IDs: "
        f"{missing_targets:,}"
    )

    # --------------------------------------------------------
    # Deduplicate predictions
    # --------------------------------------------------------

    print()
    print(
        "Building final output..."
    )

    output_rows = []

    for source1_id in source1.index:

        matches = predictions.get(
            source1_id,
            [],
        )

        matches = list(
            dict.fromkeys(
                matches
            )
        )

        output_rows.append(
            (
                source1_id,
                ",".join(matches),
            )
        )

    result = pd.DataFrame(
        output_rows,
        columns=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
    )

    empty_count = int(
        (
            result[
                "matched_entity_ids"
            ]
            == ""
        ).sum()
    )

    nonempty_count = (
        len(result)
        - empty_count
    )

    total_links = int(
        result[
            "matched_entity_ids"
        ]
        .map(
            lambda x:
                len(
                    [
                        y
                        for y
                        in str(x).split(",")
                        if y
                    ]
                )
        )
        .sum()
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "ML INFERENCE COMPLETE"
    )
    print("=" * 70)

    print(
        f"Output: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Source 1 rows: "
        f"{len(result):,}"
    )

    print(
        f"Non-empty matches: "
        f"{nonempty_count:,}"
    )

    print(
        f"Empty matches: "
        f"{empty_count:,}"
    )

    print(
        f"Total predicted links: "
        f"{total_links:,}"
    )

    print(
        f"Average links / S1: "
        f"{total_links / len(result):.4f}"
    )

    print(
        f"Threshold used: "
        f"{threshold:.4f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()