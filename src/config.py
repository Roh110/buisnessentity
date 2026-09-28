from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "extracted" / "student_resource"
DATASET_ROOT = DATA_ROOT / "dataset"

TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"


# ============================================================
# Training files
# ============================================================

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"


# ============================================================
# Test files
# ============================================================

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"


# ============================================================
# Output directories
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "models"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"


# ============================================================
# Dataset schema
# ============================================================

ENTITY_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

GROUND_TRUTH_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]

CANDIDATE_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_ids",
]

MATCHING_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]


# ============================================================
# File format
# ============================================================

TSV_SEPARATOR = "\t"
FILE_ENCODING = "utf-8"