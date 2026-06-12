from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.db.config import get_database_settings, require_database_url


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def create_database_engine(database_url: str | None = None) -> Engine:
    settings = get_database_settings()
    resolved_database_url = database_url or require_database_url(settings)
    return create_engine(
        resolved_database_url,
        echo=settings.echo,
        pool_pre_ping=settings.pool_pre_ping,
    )


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_database_engine()
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
    return _session_factory


def get_db_session() -> Generator[Session, None, None]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
