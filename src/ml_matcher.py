from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

import joblib
import pandas as pd
from rapidfuzz import fuzz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    OUTPUT_DIR,
)
from src.normalization import (
    normalize_business_name,
    normalize_address,
    normalize_country,
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

SAMPLE_S1 = 80_000

MAX_POSITIVES = 100_000
MAX_NEGATIVES = 200_000

CHUNK_SIZE = 500_000


FEATURE_NAMES = [
    "name_exact",
    "address_exact",
    "country_exact",
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_jaccard",
    "address_ratio",
    "address_token_sort",
    "address_token_set",
    "address_jaccard",
]


# ============================================================
# TEXT HELPERS
# ============================================================

def tokenize(value: str) -> Set[str]:
    if not value:
        return set()

    return set(value.split())


def jaccard(a: str, b: str) -> float:
    if not a or not b:
        return 0.0

    a_tokens = tokenize(a)
    b_tokens = tokenize(b)

    union = a_tokens | b_tokens

    if not union:
        return 0.0

    return len(a_tokens & b_tokens) / len(union)


# ============================================================
# FEATURE GENERATION
# ============================================================

def make_features(
    name_a: str,
    address_a: str,
    country_a: str,
    name_b: str,
    address_b: str,
    country_b: str,
) -> List[float]:

    return [
        # Exact evidence
        float(
            bool(name_a)
            and name_a == name_b
        ),

        float(
            bool(address_a)
            and address_a == address_b
        ),

        float(
            bool(country_a)
            and country_a == country_b
        ),

        # Name similarity
        fuzz.ratio(
            name_a,
            name_b,
        ) / 100.0,

        fuzz.token_sort_ratio(
            name_a,
            name_b,
        ) / 100.0,

        fuzz.token_set_ratio(
            name_a,
            name_b,
        ) / 100.0,

        jaccard(
            name_a,
            name_b,
        ),

        # Address similarity
        fuzz.ratio(
            address_a,
            address_b,
        ) / 100.0,

        fuzz.token_sort_ratio(
            address_a,
            address_b,
        ) / 100.0,

        fuzz.token_set_ratio(
            address_a,
            address_b,
        ) / 100.0,

        jaccard(
            address_a,
            address_b,
        ),
    ]


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth() -> Dict[str, Set[str]]:

    print("Loading ground truth...")

    df = pd.read_csv(
        TRAIN_GROUND_TRUTH,
        sep="\t",
        dtype="string",
        keep_default_na=False,
    )

    result: Dict[str, Set[str]] = {}

    for row in df.itertuples(index=False):

        source1_id = str(
            row.source1_entity_id
        )

        raw = str(
            row.matched_entity_ids
        ).strip()

        if raw:

            result[source1_id] = {
                x.strip()
                for x in raw.split(",")
                if x.strip()
            }

        else:

            result[source1_id] = set()

    print(
        f"Ground-truth rows: "
        f"{len(result):,}"
    )

    return result


# ============================================================
# SAMPLE SOURCE 1
# ============================================================

def sample_source1(
    ground_truth: Dict[str, Set[str]],
) -> Tuple[
    pd.DataFrame,
    Dict[str, Set[str]],
]:

    print()
    print(
        f"Sampling {SAMPLE_S1:,} "
        f"Source-1 entities..."
    )

    rng = random.Random(
        RANDOM_STATE
    )

    all_ids = list(
        ground_truth.keys()
    )

    sampled_ids = set(
        rng.sample(
            all_ids,
            min(
                SAMPLE_S1,
                len(all_ids),
            ),
        )
    )

    rows = []

    for chunk in pd.read_csv(
        TRAIN_SOURCE1,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):

        selected = chunk[
            chunk["entity_id"].isin(
                sampled_ids
            )
        ]

        if not selected.empty:
            rows.append(selected)

        if sum(
            len(x)
            for x in rows
        ) >= len(sampled_ids):

            break

    if not rows:
        raise RuntimeError(
            "Could not find sampled Source-1 records."
        )

    source1 = pd.concat(
        rows,
        ignore_index=True,
    )

    sampled_ground_truth = {
        entity_id: ground_truth[entity_id]
        for entity_id in source1["entity_id"]
        if entity_id in ground_truth
    }

    print(
        f"Sampled Source-1 rows: "
        f"{len(source1):,}"
    )

    return (
        source1,
        sampled_ground_truth,
    )


# ============================================================
# PREPARE SOURCE 1
# ============================================================

def prepare_source1(
    source1: pd.DataFrame,
) -> pd.DataFrame:

    source1 = source1.copy()

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

    return source1


# ============================================================
# COLLECT TARGET IDS
# ============================================================

def collect_required_target_ids(
    ground_truth: Dict[str, Set[str]],
) -> Set[str]:

    required: Set[str] = set()

    for matches in ground_truth.values():
        required.update(matches)

    print(
        f"Unique positive target IDs needed: "
        f"{len(required):,}"
    )

    return required


# ============================================================
# LOAD REQUIRED TARGET RECORDS
# ============================================================

def load_required_targets(
    required_ids: Set[str],
) -> pd.DataFrame:

    print()
    print(
        "Scanning target files for "
        "positive examples..."
    )

    pieces = []

    for path in [
        TRAIN_SOURCE2,
        TRAIN_SOURCE3,
    ]:

        print(
            f"Reading: {path.name}"
        )

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype="string",
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
        ):

            selected = chunk[
                chunk["entity_id"].isin(
                    required_ids
                )
            ]

            if not selected.empty:
                pieces.append(selected)

    if not pieces:
        raise RuntimeError(
            "Could not find ground-truth target IDs."
        )

    targets = pd.concat(
        pieces,
        ignore_index=True,
    )

    targets["name_norm"] = (
        targets["business_name"]
        .astype(str)
        .map(normalize_business_name)
    )

    targets["address_norm"] = (
        targets["business_address"]
        .astype(str)
        .map(normalize_address)
    )

    targets["country_norm"] = (
        targets["country"]
        .astype(str)
        .map(normalize_country)
    )

    print(
        f"Required target records loaded: "
        f"{len(targets):,}"
    )

    return targets


# ============================================================
# BUILD TRAINING PAIRS
# ============================================================

def build_training_pairs(
    source1: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    targets: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
]:

    print()
    print(
        "Building supervised training pairs..."
    )

    target_by_id = targets.set_index(
        "entity_id"
    )

    name_index: Dict[
        Tuple[str, str],
        List[str],
    ] = {}

    address_index: Dict[
        Tuple[str, str],
        List[str],
    ] = {}

    # --------------------------------------------------------
    # Target blocking indexes
    # --------------------------------------------------------

    for row in targets.itertuples(
        index=False
    ):

        entity_id = str(
            row.entity_id
        )

        country = str(
            row.country_norm
        )

        name = str(
            row.name_norm
        )

        address = str(
            row.address_norm
        )

        if name:

            name_index.setdefault(
                (country, name),
                [],
            ).append(entity_id)

        if address:

            address_index.setdefault(
                (country, address),
                [],
            ).append(entity_id)

    positives = []
    negatives = []

    rng = random.Random(
        RANDOM_STATE
    )

    # --------------------------------------------------------
    # Generate examples
    # --------------------------------------------------------

    for row in source1.itertuples(
        index=False
    ):

        source1_id = str(
            row.entity_id
        )

        true_ids = ground_truth.get(
            source1_id,
            set(),
        )

        country = str(
            row.country_norm
        )

        name = str(
            row.name_norm
        )

        address = str(
            row.address_norm
        )

        candidate_ids: Set[str] = set()

        # Same country + exact name
        if name:

            candidate_ids.update(
                name_index.get(
                    (country, name),
                    [],
                )
            )

        # Same country + exact address
        if address:

            candidate_ids.update(
                address_index.get(
                    (country, address),
                    [],
                )
            )

        # Always include known positives
        candidate_ids.update(
            true_ids
        )

        for candidate_id in candidate_ids:

            if candidate_id not in target_by_id.index:
                continue

            target = target_by_id.loc[
                candidate_id
            ]

            features = make_features(
                name,
                address,
                country,
                str(target["name_norm"]),
                str(target["address_norm"]),
                str(target["country_norm"]),
            )

            if candidate_id in true_ids:

                if (
                    len(positives)
                    < MAX_POSITIVES
                ):

                    positives.append(
                        features
                    )

            else:

                if (
                    len(negatives)
                    < MAX_NEGATIVES
                ):

                    negatives.append(
                        features
                    )

        if (
            len(positives)
            >= MAX_POSITIVES
            and
            len(negatives)
            >= MAX_NEGATIVES
        ):
            break

    rng.shuffle(
        negatives
    )

    negatives = negatives[
        :MAX_NEGATIVES
    ]

    rows = (
        positives
        + negatives
    )

    labels = (
        [1] * len(positives)
        +
        [0] * len(negatives)
    )

    X = pd.DataFrame(
        rows,
        columns=FEATURE_NAMES,
    )

    y = pd.Series(
        labels,
        dtype="int8",
        name="label",
    )

    print(
        f"Positive pairs: "
        f"{len(positives):,}"
    )

    print(
        f"Negative pairs: "
        f"{len(negatives):,}"
    )

    print(
        f"Total pairs   : "
        f"{len(X):,}"
    )

    return X, y


# ============================================================
# TRAIN + CONSERVATIVE THRESHOLD TUNING
# ============================================================

def train_and_tune(
    X: pd.DataFrame,
    y: pd.Series,
):

    print()
    print(
        "Splitting training/validation data..."
    )

    X_train, X_valid, y_train, y_valid = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    print(
        f"Train rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(X_valid):,}"
    )

    print()
    print(
        "Training classifier..."
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=2.0,
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Classifier trained."
    )

    probabilities = model.predict_proba(
        X_valid
    )[:, 1]

    best_threshold = 0.90
    best_score = -1.0

    print()
    print(
        "Tuning conservative "
        "thresholds using validation F0.5..."
    )

    thresholds = [
        0.75,
        0.80,
        0.82,
        0.84,
        0.86,
        0.88,
        0.90,
        0.92,
        0.94,
        0.96,
        0.98,
        0.99,
    ]

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype("int8")

        score = float(
            fbeta_score(
                y_valid,
                predictions,
                beta=0.5,
                zero_division=0,
            )
        )

        predicted_positive = int(
            predictions.sum()
        )

        print(
            f"threshold={threshold:.2f} "
            f"F0.5={score:.6f} "
            f"predicted={predicted_positive:,}"
        )

        if score > best_score:

            best_score = score
            best_threshold = float(
                threshold
            )

    print()
    print(
        f"Selected threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation F0.5: "
        f"{best_score:.6f}"
    )

    return (
        model,
        best_threshold,
        best_score,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print(
        "BUSINESS ENTITY RESOLUTION — "
        "CONSERVATIVE ML TRAINING"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ground_truth = (
        load_ground_truth()
    )

    # --------------------------------------------------------
    # Source 1 sample
    # --------------------------------------------------------

    source1, sampled_ground_truth = (
        sample_source1(
            ground_truth
        )
    )

    source1 = prepare_source1(
        source1
    )

    # --------------------------------------------------------
    # Target IDs
    # --------------------------------------------------------

    required_target_ids = (
        collect_required_target_ids(
            sampled_ground_truth
        )
    )

    targets = (
        load_required_targets(
            required_target_ids
        )
    )

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    X, y = build_training_pairs(
        source1,
        sampled_ground_truth,
        targets,
    )

    if X.empty:
        raise RuntimeError(
            "No training pairs were generated."
        )

    if y.nunique() < 2:
        raise RuntimeError(
            "Training data contains only one class."
        )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model, threshold, score = (
        train_and_tune(
            X,
            y,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        OUTPUT_DIR
        / "entity_matcher.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "threshold": threshold,
            "features": FEATURE_NAMES,
        },
        model_path,
    )

    print()
    print("=" * 70)
    print(
        "ML TRAINING COMPLETE"
    )
    print("=" * 70)

    print(
        f"Validation F0.5 : "
        f"{score:.6f}"
    )

    print(
        f"Best threshold  : "
        f"{threshold:.2f}"
    )

    print(
        f"Model saved     : "
        f"{model_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()