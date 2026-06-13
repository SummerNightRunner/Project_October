import csv
from datetime import datetime

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from backend.app.db.models import MovieCatalogEntry
from backend.app.db.movie_catalog_sync import (
    parse_optional_date,
    read_movie_catalog_entries,
    sync_movie_catalog_entries,
)
from backend.app.db.sync_movie_catalog import main as sync_movie_catalog_main


def write_processed_metadata(path, rows, fieldnames=None):
    resolved_fieldnames = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=resolved_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_test_session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    MovieCatalogEntry.__table__.create(engine)
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    return session_factory()


def test_parse_optional_date_returns_none_for_invalid_values():
    assert parse_optional_date("") is None
    assert parse_optional_date("not-a-date") is None
    assert parse_optional_date("1995-10-30") is not None
    assert parse_optional_date("1995-10-30T12:00:00Z") is not None


def test_read_movie_catalog_entries_from_fixture_csv(tmp_path):
    metadata_path = tmp_path / "processed_metadata.csv"
    write_processed_metadata(
        metadata_path,
        [
            {
                "id": "862",
                "original_title": "Toy Story",
                "release_date": "1995-10-30",
            },
            {
                "id": "8844",
                "original_title": "Jumanji",
                "release_date": "invalid",
            },
            {
                "id": "",
                "original_title": "Missing ID",
                "release_date": "1995-01-01",
            },
        ],
    )

    entries, scanned_rows, skipped_rows = read_movie_catalog_entries(
        metadata_path=metadata_path,
        source_catalog_version="fixture-v1",
    )

    assert scanned_rows == 3
    assert skipped_rows == 1
    assert [entry.catalog_movie_id for entry in entries] == ["862", "8844"]
    assert entries[0].title_snapshot == "Toy Story"
    assert entries[0].release_date is not None
    assert entries[0].source_catalog_version == "fixture-v1"
    assert entries[1].release_date is None


def test_sync_movie_catalog_entries_upserts_without_deleting_missing_rows(tmp_path):
    first_metadata_path = tmp_path / "processed_metadata_first.csv"
    second_metadata_path = tmp_path / "processed_metadata_second.csv"
    write_processed_metadata(
        first_metadata_path,
        [
            {
                "id": "862",
                "original_title": "Toy Story",
                "release_date": "1995-10-30",
            },
            {
                "id": "8844",
                "original_title": "Jumanji",
                "release_date": "1995-12-15",
            },
        ],
    )
    write_processed_metadata(
        second_metadata_path,
        [
            {
                "id": "862",
                "original_title": "Toy Story Updated",
                "release_date": "1995-10-31",
            },
            {
                "id": "15602",
                "original_title": "Grumpier Old Men",
                "release_date": "",
            },
        ],
    )
    session = create_test_session()

    first_result = sync_movie_catalog_entries(
        session=session,
        metadata_path=first_metadata_path,
        source_catalog_version="fixture-v1",
    )
    old_updated_at = session.get(MovieCatalogEntry, "862").updated_at
    second_result = sync_movie_catalog_entries(
        session=session,
        metadata_path=second_metadata_path,
        source_catalog_version="fixture-v2",
    )

    assert first_result.inserted == 2
    assert first_result.updated == 0
    assert second_result.inserted == 1
    assert second_result.updated == 1

    rows = session.scalars(
        select(MovieCatalogEntry).order_by(MovieCatalogEntry.catalog_movie_id)
    ).all()
    assert [row.catalog_movie_id for row in rows] == ["15602", "862", "8844"]

    updated_toy_story = session.get(MovieCatalogEntry, "862")
    assert updated_toy_story.title_snapshot == "Toy Story Updated"
    assert updated_toy_story.release_date.isoformat() == "1995-10-31"
    assert updated_toy_story.source_catalog_version == "fixture-v2"
    assert updated_toy_story.created_at <= old_updated_at
    assert updated_toy_story.updated_at >= old_updated_at

    stale_jumanji = session.get(MovieCatalogEntry, "8844")
    assert stale_jumanji.title_snapshot == "Jumanji"
    assert stale_jumanji.source_catalog_version == "fixture-v1"


def test_sync_movie_catalog_cli_entrypoint_with_test_database(tmp_path):
    metadata_path = tmp_path / "processed_metadata.csv"
    database_path = tmp_path / "catalog_sync.sqlite"
    write_processed_metadata(
        metadata_path,
        [
            {
                "id": "862",
                "original_title": "Toy Story",
                "release_date": "1995-10-30",
            }
        ],
    )

    engine = create_engine(f"sqlite+pysqlite:///{database_path}")
    MovieCatalogEntry.__table__.create(engine)

    exit_code = sync_movie_catalog_main(
        [
            "--metadata-path",
            str(metadata_path),
            "--database-url",
            f"sqlite+pysqlite:///{database_path}",
            "--source-catalog-version",
            "cli-fixture",
        ]
    )

    assert exit_code == 0
    with sessionmaker(bind=engine)() as session:
        entry = session.get(MovieCatalogEntry, "862")
        assert entry is not None
        assert entry.title_snapshot == "Toy Story"
        assert entry.source_catalog_version == "cli-fixture"
        assert isinstance(entry.created_at, datetime)
