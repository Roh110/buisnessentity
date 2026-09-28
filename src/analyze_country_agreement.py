from __future__ import annotations

import sqlite3
from pathlib import Path

from src.config import (
    OUTPUT_DIR,
    TRAIN_GROUND_TRUTH,
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
)
from src.data_loader import iter_ground_truth, iter_entities


DB_PATH = OUTPUT_DIR / "country_lookup.sqlite"


def build_country_database() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        DB_PATH.unlink()

    connection = sqlite3.connect(DB_PATH)

    try:
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")

        connection.execute(
            """
            CREATE TABLE entity_country (
                entity_id TEXT PRIMARY KEY,
                country TEXT NOT NULL
            )
            """
        )

        for source_name, source_path in [
            ("S1", TRAIN_SOURCE1),
            ("S2", TRAIN_SOURCE2),
            ("S3", TRAIN_SOURCE3),
        ]:
            print(f"\nLoading {source_name} into SQLite...")

            rows_inserted = 0

            for chunk in iter_entities(source_path, chunk_size=100_000):
                rows = [
                    (str(entity_id), str(country))
                    for entity_id, country in zip(
                        chunk["entity_id"],
                        chunk["country"],
                    )
                ]

                connection.executemany(
                    """
                    INSERT INTO entity_country
                    (entity_id, country)
                    VALUES (?, ?)
                    """,
                    rows,
                )

                rows_inserted += len(rows)

                if rows_inserted % 1_000_000 < len(rows):
                    print(f"  inserted: {rows_inserted:,}")

            connection.commit()
            print(f"  {source_name} complete: {rows_inserted:,}")

    finally:
        connection.close()


def analyze_country_agreement() -> None:
    connection = sqlite3.connect(DB_PATH)

    try:
        cursor = connection.cursor()

        total_links = 0
        same_country = 0
        different_country = 0
        missing_country = 0

        print("\nAnalyzing ground truth...")

        for row_number, row in enumerate(
            iter_ground_truth(TRAIN_GROUND_TRUTH),
            start=1,
        ):
            source1_id = str(row["source1_entity_id"])
            target_ids = row["matched_entity_ids"]

            source1_result = cursor.execute(
                """
                SELECT country
                FROM entity_country
                WHERE entity_id = ?
                """,
                (source1_id,),
            ).fetchone()

            if source1_result is None:
                continue

            source1_country = source1_result[0]

            for target_id in target_ids:
                target_result = cursor.execute(
                    """
                    SELECT country
                    FROM entity_country
                    WHERE entity_id = ?
                    """,
                    (target_id,),
                ).fetchone()

                total_links += 1

                if target_result is None:
                    missing_country += 1
                elif target_result[0] == source1_country:
                    same_country += 1
                else:
                    different_country += 1

            if row_number % 250_000 == 0:
                print(
                    f"  processed Source 1 rows: {row_number:,}"
                )

        print("\n" + "=" * 60)
        print("COUNTRY AGREEMENT ANALYSIS")
        print("=" * 60)

        print(f"Total matched links : {total_links:,}")
        print(f"Same country        : {same_country:,}")
        print(f"Different country   : {different_country:,}")
        print(f"Missing country     : {missing_country:,}")

        if total_links:
            print(
                f"\nSame-country rate   : "
                f"{same_country / total_links:.4%}"
            )

    finally:
        connection.close()


def main() -> None:
    build_country_database()
    analyze_country_agreement()


if __name__ == "__main__":
    main()