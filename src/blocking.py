import pandas as pd


def find_column(df, possible_names):
    """
    Find the first matching column from a list of possible names.
    """
    for name in possible_names:
        if name in df.columns:
            return name

    return None


def get_name_column(df):
    return find_column(
        df,
        [
            "name",
            "business_name",
            "company_name",
            "entity_name",
        ],
    )


def get_address_column(df):
    return find_column(
        df,
        [
            "address",
            "full_address",
            "street_address",
        ],
    )


def get_country_column(df):
    return find_column(
        df,
        [
            "country",
            "country_code",
        ],
    )


def create_blocking_keys(df):
    """
    Create simple blocking keys for a dataframe.
    """
    df = df.copy()

    name_col = get_name_column(df)
    address_col = get_address_column(df)
    country_col = get_country_column(df)

    if name_col:
        df["_block_name"] = (
            df[name_col]
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9]", "", regex=True)
            .str[:4]
        )
    else:
        df["_block_name"] = ""

    if address_col:
        df["_block_address"] = (
            df[address_col]
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9]", "", regex=True)
            .str[:6]
        )
    else:
        df["_block_address"] = ""

    if country_col:
        df["_block_country"] = (
            df[country_col]
            .astype(str)
            .str.lower()
            .str.strip()
        )
    else:
        df["_block_country"] = ""

    return df


def generate_candidates(source1, target_df):
    """
    Generate candidate pairs between Source 1 and a target source.

    Multiple blocking rules are used and their results are combined.
    """

    source1 = create_blocking_keys(source1)
    target_df = create_blocking_keys(target_df)

    candidates = set()

    # ---------------------------------------------------------
    # Rule 1: Country + Name
    # ---------------------------------------------------------
    if "_block_country" in source1.columns:
        target_groups = {}

        for target_index, row in target_df.iterrows():
            key = (
                row["_block_country"],
                row["_block_name"],
            )

            if key not in target_groups:
                target_groups[key] = []

            target_groups[key].append(target_index)

        for source_index, row in source1.iterrows():
            key = (
                row["_block_country"],
                row["_block_name"],
            )

            for target_index in target_groups.get(key, []):
                candidates.add(
                    (source_index, target_index)
                )

    # ---------------------------------------------------------
    # Rule 2: Country + Address
    # ---------------------------------------------------------
    target_groups = {}

    for target_index, row in target_df.iterrows():
        key = (
            row["_block_country"],
            row["_block_address"],
        )

        if key not in target_groups:
            target_groups[key] = []

        target_groups[key].append(target_index)

    for source_index, row in source1.iterrows():
        key = (
            row["_block_country"],
            row["_block_address"],
        )

        for target_index in target_groups.get(key, []):
            candidates.add(
                (source_index, target_index)
            )

    # ---------------------------------------------------------
    # Rule 3: Name only
    # ---------------------------------------------------------
    target_groups = {}

    for target_index, row in target_df.iterrows():
        key = row["_block_name"]

        if key:
            if key not in target_groups:
                target_groups[key] = []

            target_groups[key].append(target_index)

    for source_index, row in source1.iterrows():
        key = row["_block_name"]

        if key:
            for target_index in target_groups.get(key, []):
                candidates.add(
                    (source_index, target_index)
                )

    # ---------------------------------------------------------
    # Rule 4: Address only
    # ---------------------------------------------------------
    target_groups = {}

    for target_index, row in target_df.iterrows():
        key = row["_block_address"]

        if key:
            if key not in target_groups:
                target_groups[key] = []

            target_groups[key].append(target_index)

    for source_index, row in source1.iterrows():
        key = row["_block_address"]

        if key:
            for target_index in target_groups.get(key, []):
                candidates.add(
                    (source_index, target_index)
                )

    # Convert to DataFrame
    candidate_df = pd.DataFrame(
        list(candidates),
        columns=[
            "source1_index",
            "target_index",
        ],
    )

    return candidate_df