from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from backend.app.db.movie_catalog_sync import (
    MovieCatalogSyncResult,
    sync_movie_catalog_entries,
)
from backend.app.db.session import create_database_engine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synchronize movie_catalog_entries from processed_metadata.csv."
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help=(
            "Path to processed_metadata.csv. Defaults to "
            "PROJECT_OCTOBER_PROCESSED_METADATA or data/processed/processed_metadata.csv."
        ),
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help=(
            "SQLAlchemy database URL. Defaults to PROJECT_OCTOBER_DATABASE_URL. "
            "Do not pass production credentials in shell history."
        ),
    )
    parser.add_argument(
        "--source-catalog-version",
        default=None,
        help="Optional stable source catalog version label to store on synced rows.",
    )
    return parser


def run_sync(
    session: Session,
    metadata_path: Path | None = None,
    source_catalog_version: str | None = None,
) -> MovieCatalogSyncResult:
    return sync_movie_catalog_entries(
        session=session,
        metadata_path=metadata_path,
        source_catalog_version=source_catalog_version,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = create_database_engine(args.database_url)
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    with session_factory() as session:
        result = run_sync(
            session=session,
            metadata_path=args.metadata_path,
            source_catalog_version=args.source_catalog_version,
        )

    print(
        "Synced movie_catalog_entries: "
        f"path={result.metadata_path} "
        f"scanned={result.scanned_rows} "
        f"skipped={result.skipped_rows} "
        f"entries={result.catalog_entries} "
        f"inserted={result.inserted} "
        f"updated={result.updated}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
