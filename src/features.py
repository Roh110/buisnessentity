from rapidfuzz import fuzz
import pandas as pd


def safe_text(value):
    """Convert a value to clean text."""
    if value is None:
        return ""

    return str(value).strip().lower()


def get_column(df, possible_names):
    """Find the first available column."""
    for name in possible_names:
        if name in df.columns:
            return name

    return None


def text_similarity(value1, value2):
    """
    Calculate several fuzzy matching scores.
    """
    a = safe_text(value1)
    b = safe_text(value2)

    if not a or not b:
        return {
            "ratio": 0.0,
            "partial_ratio": 0.0,
            "token_sort_ratio": 0.0,
            "token_set_ratio": 0.0,
        }

    return {
        "ratio": fuzz.ratio(a, b) / 100.0,
        "partial_ratio": fuzz.partial_ratio(a, b) / 100.0,
        "token_sort_ratio": fuzz.token_sort_ratio(a, b) / 100.0,
        "token_set_ratio": fuzz.token_set_ratio(a, b) / 100.0,
    }


def length_similarity(value1, value2):
    """Compare the lengths of two strings."""
    a = safe_text(value1)
    b = safe_text(value2)

    if not a or not b:
        return 0.0

    longest = max(len(a), len(b))

    return min(len(a), len(b)) / longest


def jaccard_similarity(value1, value2):
    """Calculate token-level Jaccard similarity."""
    a = set(safe_text(value1).split())
    b = set(safe_text(value2).split())

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    intersection = len(a.intersection(b))
    union = len(a.union(b))

    return intersection / union


def exact_match(value1, value2):
    """Return 1 if two non-empty values match exactly."""
    a = safe_text(value1)
    b = safe_text(value2)

    if not a or not b:
        return 0.0

    return float(a == b)


def build_pair_features(source_row, target_row):
    """
    Build similarity features for one Source 1 / target pair.
    """

    name_col_source = get_column(
        source_row.index.to_frame().T,
        [
            "name",
            "business_name",
            "company_name",
            "entity_name",
        ],
    )

    name_col_target = get_column(
        target_row.index.to_frame().T,
        [
            "name",
            "business_name",
            "company_name",
            "entity_name",
        ],
    )

    address_col_source = get_column(
        source_row.index.to_frame().T,
        [
            "address",
            "full_address",
            "street_address",
        ],
    )

    address_col_target = get_column(
        target_row.index.to_frame().T,
        [
            "address",
            "full_address",
            "street_address",
        ],
    )

    country_col_source = get_column(
        source_row.index.to_frame().T,
        [
            "country",
            "country_code",
        ],
    )

    country_col_target = get_column(
        target_row.index.to_frame().T,
        [
            "country",
            "country_code",
        ],
    )

    source_name = (
        source_row[name_col_source]
        if name_col_source
        else ""
    )

    target_name = (
        target_row[name_col_target]
        if name_col_target
        else ""
    )

    source_address = (
        source_row[address_col_source]
        if address_col_source
        else ""
    )

    target_address = (
        target_row[address_col_target]
        if address_col_target
        else ""
    )

    source_country = (
        source_row[country_col_source]
        if country_col_source
        else ""
    )

    target_country = (
        target_row[country_col_target]
        if country_col_target
        else ""
    )

    name_scores = text_similarity(
        source_name,
        target_name,
    )

    address_scores = text_similarity(
        source_address,
        target_address,
    )

    features = {
        "name_ratio": name_scores["ratio"],
        "name_partial_ratio": name_scores["partial_ratio"],
        "name_token_sort_ratio": name_scores["token_sort_ratio"],
        "name_token_set_ratio": name_scores["token_set_ratio"],

        "address_ratio": address_scores["ratio"],
        "address_partial_ratio": address_scores["partial_ratio"],
        "address_token_sort_ratio": address_scores["token_sort_ratio"],
        "address_token_set_ratio": address_scores["token_set_ratio"],

        "name_jaccard": jaccard_similarity(
            source_name,
            target_name,
        ),

        "address_jaccard": jaccard_similarity(
            source_address,
            target_address,
        ),

        "name_length_similarity": length_similarity(
            source_name,
            target_name,
        ),

        "address_length_similarity": length_similarity(
            source_address,
            target_address,
        ),

        "country_exact": exact_match(
            source_country,
            target_country,
        ),
    }

    return features


def build_feature_matrix(source1, target_df, candidate_pairs):
    """
    Create a feature matrix for all candidate pairs.
    """

    rows = []

    for _, pair in candidate_pairs.iterrows():
        source_index = pair["source1_index"]
        target_index = pair["target_index"]

        source_row = source1.loc[source_index]
        target_row = target_df.loc[target_index]

        features = build_pair_features(
            source_row,
            target_row,
        )

        features["source1_index"] = source_index
        features["target_index"] = target_index

        rows.append(features)

    return pd.DataFrame(rows)
