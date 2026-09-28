import re
import unicodedata


# Common business-name abbreviations
ABBREVIATIONS = {
    "ltd": "limited",
    "co": "company",
    "corp": "corporation",
    "inc": "incorporated",
    "pvt": "private",
    "llc": "limited liability company",
}


def normalize_text(value) -> str:
    """
    General text normalization.

    Steps:
    1. Convert to string
    2. Convert to lowercase
    3. Remove accents
    4. Replace punctuation with spaces
    5. Normalize whitespace
    """
    if value is None:
        return ""

    value = str(value).strip().lower()

    # Remove Unicode accents
    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
    )

    # Replace punctuation/symbols with spaces
    value = re.sub(r"[^\w\s]", " ", value)

    # Normalize whitespace
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_business_name(value) -> str:
    """Normalize a business name."""
    value = normalize_text(value)

    words = value.split()

    normalized_words = [
        ABBREVIATIONS.get(word, word)
        for word in words
    ]

    return " ".join(normalized_words)


def normalize_address(value) -> str:
    """Normalize a business address."""
    return normalize_text(value)


def normalize_country(value) -> str:
    """Normalize country names/codes."""
    return normalize_text(value)


def add_normalized_columns(df):
    """
    Add normalized versions of common entity fields.

    Only columns that actually exist are processed.
    """
    df = df.copy()

    possible_name_columns = [
        "name",
        "business_name",
        "company_name",
        "entity_name",
    ]

    possible_address_columns = [
        "address",
        "full_address",
        "street_address",
    ]

    possible_country_columns = [
        "country",
        "country_code",
    ]

    for column in possible_name_columns:
        if column in df.columns:
            df[f"{column}_norm"] = df[column].apply(
                normalize_business_name
            )

    for column in possible_address_columns:
        if column in df.columns:
            df[f"{column}_norm"] = df[column].apply(
                normalize_address
            )

    for column in possible_country_columns:
        if column in df.columns:
            df[f"{column}_norm"] = df[column].apply(
                normalize_country
            )

    return df