from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from backend.app.catalog_paths import get_processed_metadata_path
from backend.app.db.models import MovieCatalogEntry


TITLE_COLUMN_CANDIDATES = ("original_title", "title", "name")
RELEASE_DATE_COLUMN_CANDIDATES = ("release_date",)


@dataclass(frozen=True)
class MovieCatalogSyncEntry:
    catalog_movie_id: str
    title_snapshot: str | None
    release_date: date | None
    source_catalog_version: str | None


@dataclass(frozen=True)
class MovieCatalogSyncResult:
    metadata_path: Path
    scanned_rows: int
    skipped_rows: int
    catalog_entries: int
    inserted: int
    updated: int


def parse_optional_date(value: Any) -> date | None:
    if value is None:
        return None

    normalized_value = str(value).strip()
    if not normalized_value:
        return None

    try:
        return date.fromisoformat(normalized_value[:10])
    except ValueError:
        return None


def normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None

    normalized_value = str(value).strip()
    return normalized_value or None


def find_first_existing_column(
    fieldnames: list[str] | None, candidates: tuple[str, ...]
) -> str | None:
    if not fieldnames:
        return None

    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def read_movie_catalog_entries(
    metadata_path: Path | None = None,
    source_catalog_version: str | None = None,
) -> tuple[list[MovieCatalogSyncEntry], int, int]:
    resolved_metadata_path = metadata_path or get_processed_metadata_path()
    entries_by_id: dict[str, MovieCatalogSyncEntry] = {}
    scanned_rows = 0
    skipped_rows = 0

    with resolved_metadata_path.open(newline="", encoding="utf-8") as metadata_file:
        reader = csv.DictReader(metadata_file)
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError("Movie catalog data must contain an id column.")

        title_column = find_first_existing_column(
            reader.fieldnames,
            TITLE_COLUMN_CANDIDATES,
        )
        release_date_column = find_first_existing_column(
            reader.fieldnames,
            RELEASE_DATE_COLUMN_CANDIDATES,
        )

        for row in reader:
            scanned_rows += 1
            catalog_movie_id = normalize_optional_text(row.get("id"))
            if catalog_movie_id is None:
                skipped_rows += 1
                continue

            title_snapshot = (
                normalize_optional_text(row.get(title_column)) if title_column else None
            )
            release_date = (
                parse_optional_date(row.get(release_date_column))
                if release_date_column
                else None
            )
            entries_by_id[catalog_movie_id] = MovieCatalogSyncEntry(
                catalog_movie_id=catalog_movie_id,
                title_snapshot=title_snapshot,
                release_date=release_date,
                source_catalog_version=source_catalog_version,
            )

    return list(entries_by_id.values()), scanned_rows, skipped_rows


def sync_movie_catalog_entries(
    session: Session,
    metadata_path: Path | None = None,
    source_catalog_version: str | None = None,
) -> MovieCatalogSyncResult:
    resolved_metadata_path = metadata_path or get_processed_metadata_path()
    entries, scanned_rows, skipped_rows = read_movie_catalog_entries(
        metadata_path=resolved_metadata_path,
        source_catalog_version=source_catalog_version,
    )
    synchronized_at = datetime.now(UTC)
    inserted = 0
    updated = 0

    for entry in entries:
        existing_entry = session.get(MovieCatalogEntry, entry.catalog_movie_id)
        if existing_entry is None:
            session.add(
                MovieCatalogEntry(
                    catalog_movie_id=entry.catalog_movie_id,
                    title_snapshot=entry.title_snapshot,
                    release_date=entry.release_date,
                    source_catalog_version=entry.source_catalog_version,
                    created_at=synchronized_at,
                    updated_at=synchronized_at,
                )
            )
            inserted += 1
            continue

        existing_entry.title_snapshot = entry.title_snapshot
        existing_entry.release_date = entry.release_date
        existing_entry.source_catalog_version = entry.source_catalog_version
        existing_entry.updated_at = synchronized_at
        updated += 1

    session.commit()
    return MovieCatalogSyncResult(
        metadata_path=resolved_metadata_path,
        scanned_rows=scanned_rows,
        skipped_rows=skipped_rows,
        catalog_entries=len(entries),
        inserted=inserted,
        updated=updated,
    )
