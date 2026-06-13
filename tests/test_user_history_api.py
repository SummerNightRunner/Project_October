from datetime import UTC, datetime
import uuid

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.db.models import (
    MovieCatalogEntry,
    User,
    UserMovieHistory,
    UserMovieRating,
)
from backend.app.db.session import get_db_session
from backend.app.main import app


def create_user_history_test_session_factory():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as connection:
        connection.exec_driver_sql(
            """
            CREATE TABLE users (
                id CHAR(32) NOT NULL PRIMARY KEY,
                email TEXT,
                display_name TEXT,
                status TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TABLE movie_catalog_entries (
                catalog_movie_id TEXT NOT NULL PRIMARY KEY,
                title_snapshot TEXT,
                release_date DATE,
                source_catalog_version TEXT,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TABLE user_movie_history (
                id CHAR(32) NOT NULL PRIMARY KEY,
                user_id CHAR(32) NOT NULL,
                catalog_movie_id TEXT NOT NULL,
                status TEXT NOT NULL,
                watched_at DATETIME,
                source TEXT NOT NULL,
                notes TEXT,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                UNIQUE(user_id, catalog_movie_id)
            )
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TABLE user_movie_ratings (
                id CHAR(32) NOT NULL PRIMARY KEY,
                user_id CHAR(32) NOT NULL,
                catalog_movie_id TEXT NOT NULL,
                rating_value NUMERIC(3, 1) NOT NULL,
                rated_at DATETIME,
                source TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                UNIQUE(user_id, catalog_movie_id)
            )
            """
        )

    return sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )


@pytest.fixture
def user_history_client():
    session_factory = create_user_history_test_session_factory()

    def override_get_db_session():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db_session] = override_get_db_session
    try:
        yield TestClient(app), session_factory
    finally:
        app.dependency_overrides.pop(get_db_session, None)


def add_catalog_movies(session_factory, *movie_ids: str) -> None:
    created_at = datetime.now(UTC)
    with session_factory() as session:
        session.add_all(
            [
                MovieCatalogEntry(
                    catalog_movie_id=movie_id,
                    title_snapshot=f"Movie {movie_id}",
                    created_at=created_at,
                    updated_at=created_at,
                )
                for movie_id in movie_ids
            ]
        )
        session.commit()


def test_put_user_history_creates_user_and_history(user_history_client):
    client, session_factory = user_history_client
    user_id = uuid.uuid4()
    add_catalog_movies(session_factory, "862")

    response = client.put(
        f"/users/{user_id}/history/862",
        json={
            "status": "watched",
            "watched_at": "2026-06-13T12:00:00Z",
            "source": "manual",
            "notes": "Пересмотреть позже",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "movie_id": "862",
        "status": "watched",
        "watched_at": "2026-06-13T12:00:00Z",
        "rating_value": None,
        "source": "manual",
        "notes": "Пересмотреть позже",
    }

    with session_factory() as session:
        user = session.get(User, user_id)
        history_entry = session.scalar(
            select(UserMovieHistory).where(UserMovieHistory.user_id == user_id)
        )

    assert user is not None
    assert user.status == "active"
    assert history_entry is not None
    assert history_entry.catalog_movie_id == "862"


def test_put_user_history_updates_existing_entry(user_history_client):
    client, session_factory = user_history_client
    user_id = uuid.uuid4()
    add_catalog_movies(session_factory, "862")

    first_response = client.put(
        f"/users/{user_id}/history/862",
        json={"status": "planned", "source": "manual"},
    )
    second_response = client.put(
        f"/users/{user_id}/history/862",
        json={
            "status": "dropped",
            "watched_at": None,
            "source": "api",
            "notes": "Не досмотрел",
        },
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert second_response.json()["status"] == "dropped"
    assert second_response.json()["source"] == "api"

    with session_factory() as session:
        history_entries = session.scalars(select(UserMovieHistory)).all()

    assert len(history_entries) == 1
    assert history_entries[0].status == "dropped"


def test_get_user_history_filters_by_status_and_limit(user_history_client):
    client, session_factory = user_history_client
    user_id = uuid.uuid4()
    add_catalog_movies(session_factory, "862", "8844", "15602")

    client.put(
        f"/users/{user_id}/history/862",
        json={
            "status": "watched",
            "watched_at": "2026-06-13T12:00:00Z",
            "source": "manual",
        },
    )
    client.put(
        f"/users/{user_id}/history/8844",
        json={
            "status": "planned",
            "watched_at": None,
            "source": "manual",
        },
    )
    client.put(
        f"/users/{user_id}/history/15602",
        json={
            "status": "watched",
            "watched_at": "2026-06-12T12:00:00Z",
            "source": "manual",
        },
    )

    response = client.get(
        f"/users/{user_id}/history",
        params={"status": "watched", "limit": 1},
    )

    assert response.status_code == 200
    assert response.json()["items"] == [
        {
            "movie_id": "862",
            "status": "watched",
            "watched_at": "2026-06-13T12:00:00Z",
            "rating_value": None,
            "source": "manual",
            "notes": None,
        }
    ]


def test_put_user_rating_creates_user_rating_and_history_includes_rating(
    user_history_client,
):
    client, session_factory = user_history_client
    user_id = uuid.uuid4()
    add_catalog_movies(session_factory, "862")

    history_response = client.put(
        f"/users/{user_id}/history/862",
        json={"status": "watched", "source": "manual"},
    )
    rating_response = client.put(
        f"/users/{user_id}/ratings/862",
        json={
            "rating_value": 8.5,
            "rated_at": "2026-06-13T12:05:00Z",
            "source": "manual",
        },
    )

    assert history_response.status_code == 200
    assert rating_response.status_code == 200
    assert rating_response.json() == {
        "movie_id": "862",
        "rating_value": 8.5,
        "rated_at": "2026-06-13T12:05:00Z",
        "source": "manual",
    }

    history_list_response = client.get(f"/users/{user_id}/history")

    assert history_list_response.status_code == 200
    assert history_list_response.json()["items"][0]["rating_value"] == 8.5

    with session_factory() as session:
        ratings = session.scalars(select(UserMovieRating)).all()

    assert len(ratings) == 1
    assert float(ratings[0].rating_value) == 8.5


def test_put_user_rating_updates_existing_rating(user_history_client):
    client, session_factory = user_history_client
    user_id = uuid.uuid4()
    add_catalog_movies(session_factory, "862")

    first_response = client.put(
        f"/users/{user_id}/ratings/862",
        json={"rating_value": 8.5, "source": "manual"},
    )
    second_response = client.put(
        f"/users/{user_id}/ratings/862",
        json={"rating_value": 9.0, "source": "api"},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert second_response.json()["rating_value"] == 9.0
    assert second_response.json()["source"] == "api"

    with session_factory() as session:
        ratings = session.scalars(select(UserMovieRating)).all()

    assert len(ratings) == 1
    assert float(ratings[0].rating_value) == 9.0


def test_put_user_history_returns_404_for_unknown_movie(user_history_client):
    client, _ = user_history_client
    user_id = uuid.uuid4()

    response = client.put(
        f"/users/{user_id}/history/missing",
        json={"status": "watched", "source": "manual"},
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": (
            "Movie with movie_id 'missing' was not found in "
            "movie_catalog_entries."
        )
    }


def test_put_user_rating_rejects_out_of_range_rating(user_history_client):
    client, session_factory = user_history_client
    user_id = uuid.uuid4()
    add_catalog_movies(session_factory, "862")

    response = client.put(
        f"/users/{user_id}/ratings/862",
        json={"rating_value": 10.5, "source": "manual"},
    )

    assert response.status_code == 422
