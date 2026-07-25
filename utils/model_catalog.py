"""Load the core model catalog and independently reviewable expansions."""

import json
from pathlib import Path


CATALOG_FILES = (
    "model_database.json",
    "model_expansions.json",
    "time_series_models.json",
)


def load_model_catalog(base_dir: Path | str) -> dict:
    data_dir = Path(base_dir) / "data"
    catalog = {}

    for filename in CATALOG_FILES:
        catalog_path = data_dir / filename
        if not catalog_path.is_file():
            if filename == "model_database.json":
                raise RuntimeError(f"Required model database is missing: {catalog_path}")
            continue

        with catalog_path.open(encoding="utf-8") as catalog_file:
            entries = json.load(catalog_file)

        if not isinstance(entries, dict) or not entries:
            raise RuntimeError(f"Model catalog must be a non-empty JSON object: {catalog_path}")

        duplicate_names = catalog.keys() & entries.keys()
        if duplicate_names:
            duplicates = ", ".join(sorted(duplicate_names))
            raise RuntimeError(f"Duplicate model names across catalog files: {duplicates}")

        catalog.update(entries)

    return catalog
