import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_METADATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "processed_metadata.csv"
)
PROCESSED_METADATA_ENV = "PROJECT_OCTOBER_PROCESSED_METADATA"


def get_processed_metadata_path() -> Path:
    return Path(os.environ.get(PROCESSED_METADATA_ENV, DEFAULT_PROCESSED_METADATA_PATH))
