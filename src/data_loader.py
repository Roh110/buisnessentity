from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, TypedDict
class GroundTruthRow(TypedDict):
    source1_entity_id: str
    matched_entity_ids: list[str]

import pandas as pd

from src.config import (
    ENTITY_COLUMNS,
    FILE_ENCODING,
    GROUND_TRUTH_COLUMNS,
    TSV_SEPARATOR,
)


def load_entities(
    path: Path,
    *,
    usecols: list[str] | None = None,
    dtype: str = "string",
) -> pd.DataFrame:
    """
    Load a Source 1, Source 2, or Source 3 TSV file.

    The challenge files use UTF-8 TSV format with the columns:
        entity_id
        business_name
        business_address
        country
    """
    columns = usecols or ENTITY_COLUMNS

    return pd.read_csv(
        path,
        sep=TSV_SEPARATOR,
        encoding=FILE_ENCODING,
        usecols=columns,
        dtype=dtype,
        keep_default_na=False,
    )


def iter_entities(
    path: Path,
    *,
    chunk_size: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """
    Stream an entity TSV file in chunks.

    This is the preferred loader for the very large Source 2 and
    Source 3 files.
    """
    for chunk in pd.read_csv(
        path,
        sep=TSV_SEPARATOR,
        encoding=FILE_ENCODING,
        usecols=ENTITY_COLUMNS,
        dtype="string",
        keep_default_na=False,
        chunksize=chunk_size,
    ):
        yield chunk


def load_ground_truth(path: Path) -> pd.DataFrame:
    """
    Load the one-to-many training ground truth.

    Each Source 1 entity has one row:

        source1_entity_id
        matched_entity_ids

    matched_entity_ids is a comma-separated list and may be empty.
    """
    return pd.read_csv(
        path,
        sep=TSV_SEPARATOR,
        encoding=FILE_ENCODING,
        usecols=GROUND_TRUTH_COLUMNS,
        dtype="string",
        keep_default_na=False,
    )


def iter_ground_truth(path: Path) -> Iterator[GroundTruthRow]:
    """
    Stream ground-truth rows without loading the entire file into memory.

    Each yielded record contains:
        source1_entity_id: str
        matched_entity_ids: list[str]
    """
    with path.open(
        "r",
        encoding=FILE_ENCODING,
        newline="",
    ) as file:
        reader = csv.DictReader(file, delimiter=TSV_SEPARATOR)

        for row in reader:
            raw_matches = (row["matched_entity_ids"] or "").strip()

            if raw_matches:
                matches = [
                    entity_id.strip()
                    for entity_id in raw_matches.split(",")
                    if entity_id.strip()
                ]
            else:
                matches = []

            yield {
                "source1_entity_id": row["source1_entity_id"].strip(),
                "matched_entity_ids": matches,
            }