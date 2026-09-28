from __future__ import annotations

from typing import cast

from src.config import TRAIN_GROUND_TRUTH
from src.data_loader import iter_ground_truth


def main() -> None:
    print("Testing ground-truth target IDs...")

    total_links = 0
    source2_links = 0
    source3_links = 0

    for row in iter_ground_truth(TRAIN_GROUND_TRUTH):
        target_ids = cast(
            list[str],
            row.get("matched_entity_ids", []),
        )

        for target_id in target_ids:
            total_links += 1

            if target_id.startswith("S2-"):
                source2_links += 1

            elif target_id.startswith("S3-"):
                source3_links += 1

    print("\n" + "=" * 60)
    print("GROUND TRUTH TARGET ID ANALYSIS")
    print("=" * 60)

    print(f"Total matched links : {total_links:,}")
    print(f"S2 matched links    : {source2_links:,}")
    print(f"S3 matched links    : {source3_links:,}")


if __name__ == "__main__":
    main()