from datetime import UTC, datetime, timedelta
import uuid

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.api_key_auth import (
    HISTORY_READ_SCOPE,
    HISTORY_WRITE_SCOPE,
    PREFERENCES_READ_SCOPE,
    PREFERENCES_WRITE_SCOPE,
    RATINGS_WRITE_SCOPE,
    create_api_key_hash,
    extract_api_key_prefix,
)
from backend.app.db.models import (
    ApiClient,
    ApiKey,
    MovieCatalogEntry,
    User,
    UserMovieHistory,
    UserMovieRating,
    UserPreference,
)
from backend.app.db.session import get_db_session
from backend.app.main import app


DEFAULT_API_KEY = "oct_allscopes_test-secret"
DEFAULT_OWNER_USER_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")


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
        connection.exec_driver_sql(
            """
            CREATE TABLE user_preferences (
                id CHAR(32) NOT NULL PRIMARY KEY,
                user_id CHAR(32) NOT NULL,
                preference_type TEXT NOT NULL,
                preference_key TEXT NOT NULL,
                weight NUMERIC(4, 2) NOT NULL,
                source TEXT NOT NULL,
                is_active BOOLEAN NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                UNIQUE(user_id, preference_type, preference_key)
            )
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TABLE api_clients (
                id CHAR(32) NOT NULL PRIMARY KEY,
                owner_user_id CHAR(32),
                name TEXT NOT NULL,
                contact_email TEXT,
                status TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TABLE api_keys (
                id CHAR(32) NOT NULL PRIMARY KEY,
                api_client_id CHAR(32) NOT NULL,
                key_prefix TEXT NOT NULL UNIQUE,
                key_hash TEXT NOT NULL,
                scopes JSON NOT NULL,
                status TEXT NOT NULL,
                expires_at DATETIME,
                last_used_at DATETIME,
                created_at DATETIME NOT NULL,
                revoked_at DATETIME
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
    add_api_key(
        session_factory,
        api_key=DEFAULT_API_KEY,
        scopes=[
            HISTORY_READ_SCOPE,
            HISTORY_WRITE_SCOPE,
            RATINGS_WRITE_SCOPE,
            PREFERENCES_READ_SCOPE,
            PREFERENCES_WRITE_SCOPE,
        ],
        owner_user_id=DEFAULT_OWNER_USER_ID,
    )

    def override_get_db_session():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db_session] = override_get_db_session
    try:
        yield (
            TestClient(
                app,
                headers={"Authorization": f"Bearer {DEFAULT_API_KEY}"},
            ),
            session_factory,
        )
    finally:
        app.dependency_overrides.pop(get_db_session, None)


def add_api_key(
    session_factory,
    *,
    api_key: str,
    scopes: list[str],
    owner_user_id: uuid.UUID | None = None,
    status: str = "active",
    client_status: str = "active",
    expires_at: datetime | None = None,
) -> ApiKey:
    created_at = datetime.now(UTC)
    key_prefix = extract_api_key_prefix(api_key)
    assert key_prefix is not None

    with session_factory() as session:
        if owner_user_id is not None and session.get(User, owner_user_id) is None:
            session.add(
                User(
                    id=owner_user_id,
                    status="active",
                    created_at=created_at,
                    updated_at=created_at,
                )
            )
        api_client = ApiClient(
            id=uuid.uuid4(),
            owner_user_id=owner_user_id,
            name=f"Test client {key_prefix}",
            status=client_status,
            created_at=created_at,
            updated_at=created_at,
        )
        api_key_record = ApiKey(
            id=uuid.uuid4(),
            api_client_id=api_client.id,
            key_prefix=key_prefix,
            key_hash=create_api_key_hash(api_key),
            scopes=scopes,
            status=status,
            expires_at=expires_at,
            created_at=created_at,
        )
        session.add(api_client)
        session.add(api_key_record)
        session.commit()
        return api_key_record


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


def get_api_key_record(session_factory, api_key: str) -> ApiKey:
    key_prefix = extract_api_key_prefix(api_key)
    assert key_prefix is not None
    with session_factory() as session:
        api_key_record = session.scalar(
            select(ApiKey).where(ApiKey.key_prefix == key_prefix)
        )
        assert api_key_record is not None
        return api_key_record


def test_user_history_requires_authorization_header(user_history_client):
    _, _session_factory = user_history_client
    user_id = uuid.uuid4()
    client = TestClient(app)

    response = client.get(f"/users/{user_id}/history")

    assert response.status_code == 401
    assert response.json() == {"detail": "Missing API key."}


def test_user_history_rejects_invalid_api_key(user_history_client):
    _, _session_factory = user_history_client
    user_id = uuid.uuid4()
    client = TestClient(app)

    response = client.get(
        f"/users/{user_id}/history",
        headers={"Authorization": "Bearer oct_missing_test-secret"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid API key."}


def test_user_history_does_not_update_last_used_at_for_invalid_secret(
    user_history_client,
):
    _, session_factory = user_history_client
    user_id = uuid.uuid4()
    stored_key = "oct_badhash_real-secret"
    presented_key = "oct_badhash_wrong-secret"
    add_api_key(
        session_factory,
        api_key=stored_key,
        scopes=[HISTORY_READ_SCOPE],
        owner_user_id=user_id,
    )
    client = TestClient(app)

    response = client.get(
        f"/users/{user_id}/history",
        headers={"Authorization": f"Bearer {presented_key}"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid API key."}
    api_key_record = get_api_key_record(session_factory, stored_key)
    assert api_key_record.last_used_at is None


def test_user_history_rejects_revoked_api_key(user_history_client):
    _, session_factory = user_history_client
    user_id = uuid.uuid4()
    revoked_key = "oct_revoked_test-secret"
    add_api_key(
        session_factory,
        api_key=revoked_key,
        scopes=[HISTORY_READ_SCOPE],
        status="revoked",
    )
    client = TestClient(app)

    response = client.get(
        f"/users/{user_id}/history",
        headers={"Authorization": f"Bearer {revoked_key}"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid API key."}


def test_user_history_rejects_expired_api_key(user_history_client):
    _, session_factory = user_history_client
    user_id = uuid.uuid4()
    expired_key = "oct_expired_test-secret"
    add_api_key(
        session_factory,
        api_key=expired_key,
        scopes=[HISTORY_READ_SCOPE],
        expires_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    client = TestClient(app)

    response = client.get(
        f"/users/{user_id}/history",
        headers={"Authorization": f"Bearer {expired_key}"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "API key is expired."}


def test_user_history_rejects_valid_api_key_without_required_scope(
    user_history_client,
):
    _, session_factory = user_history_client
    user_id = uuid.uuid4()
    write_only_key = "oct_writeonly_test-secret"
    add_api_key(
        session_factory,
        api_key=write_only_key,
        scopes=[HISTORY_WRITE_SCOPE],
        owner_user_id=user_id,
    )
    client = TestClient(app)

    response = client.get(
        f"/users/{user_id}/history",
        headers={"Authorization": f"Bearer {write_only_key}"},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "API key does not have the required scope."}


def test_user_history_accepts_valid_api_key_with_required_scope(
    user_history_client,
):
    _, session_factory = user_history_client
    user_id = uuid.uuid4()
    read_key = "oct_readonly_test-secret"
    add_api_key(
        session_factory,
        api_key=read_key,
        scopes=[HISTORY_READ_SCOPE],
        owner_user_id=user_id,
    )
    client = TestClient(app)

    response = client.get(
        f"/users/{user_id}/history",
        headers={"Authorization": f"Bearer {read_key}"},
    )

    assert response.status_code == 200
    assert response.json() == {"items": []}


def test_user_history_updates_last_used_at_after_success(user_history_client):
    client, session_factory = user_history_client
    user_id = DEFAULT_OWNER_USER_ID

    before_response = client.get(f"/users/{user_id}/history")

    assert before_response.status_code == 200
    api_key_record = get_api_key_record(session_factory, DEFAULT_API_KEY)
    assert api_key_record.last_used_at is not None


def test_owner_api_key_rejects_other_user_history_and_updates_last_used_at(
    user_history_client,
):
    client, session_factory = user_history_client
    other_user_id = uuid.uuid4()
    add_catalog_movies(session_factory, "862")

    read_response = client.get(f"/users/{other_user_id}/history")
    history_response = client.put(
        f"/users/{other_user_id}/history/862",
        json={"status": "watched", "source": "manual"},
    )
    rating_response = client.put(
        f"/users/{other_user_id}/ratings/862",
        json={"rating_value": 8.5, "source": "manual"},
    )

    assert read_response.status_code == 403
    assert history_response.status_code == 403
    assert rating_response.status_code == 403
    assert read_response.json() == {
        "detail": "API key is not allowed to access this user."
    }
    assert history_response.json() == read_response.json()
    assert rating_response.json() == read_response.json()
    api_key_record = get_api_key_record(session_factory, DEFAULT_API_KEY)
    assert api_key_record.last_used_at is not None

    with session_factory() as session:
        other_user = session.get(User, other_user_id)
        other_history = session.scalar(
            select(UserMovieHistory).where(UserMovieHistory.user_id == other_user_id)
        )
        other_rating = session.scalar(
            select(UserMovieRating).where(UserMovieRating.user_id == other_user_id)
        )

    assert other_user is None
    assert other_history is None
    assert other_rating is None


def test_api_client_without_owner_user_id_rejects_user_scoped_endpoint(
    user_history_client,
):
    _, session_factory = user_history_client
    user_id = uuid.uuid4()
    no_owner_key = "oct_noowner_test-secret"
    add_api_key(
        session_factory,
        api_key=no_owner_key,
        scopes=[HISTORY_READ_SCOPE],
    )
    client = TestClient(app)

    response = client.get(
        f"/users/{user_id}/history",
        headers={"Authorization": f"Bearer {no_owner_key}"},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "API key is not allowed to access this user."}
    api_key_record = get_api_key_record(session_factory, no_owner_key)
    assert api_key_record.last_used_at is not None


def test_get_user_preferences_accepts_valid_api_key_with_read_scope(
    user_history_client,
):
    _, session_factory = user_history_client
    user_id = uuid.uuid4()
    read_key = "oct_prefread_test-secret"
    add_api_key(
        session_factory,
        api_key=read_key,
        scopes=[PREFERENCES_READ_SCOPE],
        owner_user_id=user_id,
    )
    client = TestClient(app)

    response = client.get(
        f"/users/{user_id}/preferences",
        headers={"Authorization": f"Bearer {read_key}"},
    )

    assert response.status_code == 200
    assert response.json() == {"items": []}


def test_put_user_preference_creates_new_preference(user_history_client):
    client, session_factory = user_history_client
    user_id = DEFAULT_OWNER_USER_ID

    response = client.put(
        f"/users/{user_id}/preferences/genre/comedy",
        json={"weight": 2.25, "source": "manual", "is_active": True},
    )

    assert response.status_code == 200
    response_json = response.json()
    assert response_json["preference_type"] == "genre"
    assert response_json["preference_key"] == "comedy"
    assert response_json["weight"] == 2.25
    assert response_json["source"] == "manual"
    assert response_json["is_active"] is True
    assert response_json["created_at"] is not None
    assert response_json["updated_at"] is not None

    with session_factory() as session:
        user = session.get(User, user_id)
        preferences = session.scalars(select(UserPreference)).all()

    assert user is not None
    assert user.status == "active"
    assert len(preferences) == 1
    assert preferences[0].preference_key == "comedy"


def test_put_user_preference_updates_existing_preference_without_duplicate(
    user_history_client,
):
    client, session_factory = user_history_client
    user_id = DEFAULT_OWNER_USER_ID

    first_response = client.put(
        f"/users/{user_id}/preferences/keyword/space",
        json={"weight": 3.0, "source": "manual", "is_active": True},
    )
    second_response = client.put(
        f"/users/{user_id}/preferences/keyword/space",
        json={"weight": -1.5, "source": "api", "is_active": False},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert second_response.json()["weight"] == -1.5
    assert second_response.json()["source"] == "api"
    assert second_response.json()["is_active"] is False

    with session_factory() as session:
        preferences = session.scalars(select(UserPreference)).all()

    assert len(preferences) == 1
    assert float(preferences[0].weight) == -1.5
    assert preferences[0].is_active is False


def test_get_user_preferences_filters_by_active_type_and_limit(user_history_client):
    client, _session_factory = user_history_client
    user_id = DEFAULT_OWNER_USER_ID

    client.put(
        f"/users/{user_id}/preferences/genre/comedy",
        json={"weight": 2.0, "source": "manual", "is_active": True},
    )
    client.put(
        f"/users/{user_id}/preferences/keyword/space",
        json={"weight": 3.0, "source": "manual", "is_active": True},
    )
    client.put(
        f"/users/{user_id}/preferences/genre/horror",
        json={"weight": -4.0, "source": "manual", "is_active": False},
    )

    active_response = client.get(
        f"/users/{user_id}/preferences",
        params={"is_active": True, "limit": 1},
    )
    type_response = client.get(
        f"/users/{user_id}/preferences",
        params={"preference_type": "genre", "is_active": False, "limit": 20},
    )

    assert active_response.status_code == 200
    assert len(active_response.json()["items"]) == 1
    assert active_response.json()["items"][0]["is_active"] is True
    assert type_response.status_code == 200
    assert type_response.json()["items"] == [
        {
            "preference_type": "genre",
            "preference_key": "horror",
            "weight": -4.0,
            "source": "manual",
            "is_active": False,
            "created_at": type_response.json()["items"][0]["created_at"],
            "updated_at": type_response.json()["items"][0]["updated_at"],
        }
    ]


def test_put_user_preference_rejects_blank_preference_key(user_history_client):
    client, _session_factory = user_history_client
    user_id = DEFAULT_OWNER_USER_ID

    response = client.put(
        f"/users/{user_id}/preferences/genre/%20%20%20",
        json={"weight": 1.0, "source": "manual"},
    )

    assert response.status_code == 422


def test_put_user_preference_rejects_invalid_preference_type(user_history_client):
    client, _session_factory = user_history_client
    user_id = DEFAULT_OWNER_USER_ID

    response = client.put(
        f"/users/{user_id}/preferences/unsupported/comedy",
        json={"weight": 1.0, "source": "manual"},
    )

    assert response.status_code == 422


def test_put_user_preference_rejects_out_of_range_weight(user_history_client):
    client, _session_factory = user_history_client
    user_id = DEFAULT_OWNER_USER_ID

    response = client.put(
        f"/users/{user_id}/preferences/genre/comedy",
        json={"weight": 10.01, "source": "manual"},
    )

    assert response.status_code == 422


def test_put_user_preference_rejects_invalid_source(user_history_client):
    client, _session_factory = user_history_client
    user_id = DEFAULT_OWNER_USER_ID

    response = client.put(
        f"/users/{user_id}/preferences/genre/comedy",
        json={"weight": 1.0, "source": "unknown"},
    )

    assert response.status_code == 422


def test_get_user_preferences_rejects_invalid_limit(user_history_client):
    client, _session_factory = user_history_client
    user_id = DEFAULT_OWNER_USER_ID

    response = client.get(
        f"/users/{user_id}/preferences",
        params={"limit": 0},
    )

    assert response.status_code == 422


def test_user_preferences_rejects_missing_and_invalid_api_key(user_history_client):
    _, _session_factory = user_history_client
    user_id = uuid.uuid4()
    client = TestClient(app)

    missing_response = client.get(f"/users/{user_id}/preferences")
    invalid_response = client.get(
        f"/users/{user_id}/preferences",
        headers={"Authorization": "Bearer oct_missing_test-secret"},
    )

    assert missing_response.status_code == 401
    assert missing_response.json() == {"detail": "Missing API key."}
    assert invalid_response.status_code == 401
    assert invalid_response.json() == {"detail": "Invalid API key."}


def test_user_preferences_rejects_valid_api_key_without_required_scope(
    user_history_client,
):
    _, session_factory = user_history_client
    user_id = uuid.uuid4()
    history_only_key = "oct_prefnoscope_test-secret"
    add_api_key(
        session_factory,
        api_key=history_only_key,
        scopes=[HISTORY_READ_SCOPE],
        owner_user_id=user_id,
    )
    client = TestClient(app)

    read_response = client.get(
        f"/users/{user_id}/preferences",
        headers={"Authorization": f"Bearer {history_only_key}"},
    )
    write_response = client.put(
        f"/users/{user_id}/preferences/genre/comedy",
        headers={"Authorization": f"Bearer {history_only_key}"},
        json={"weight": 1.0, "source": "manual"},
    )

    assert read_response.status_code == 403
    assert read_response.json() == {
        "detail": "API key does not have the required scope."
    }
    assert write_response.status_code == 403
    assert write_response.json() == read_response.json()


def test_owner_api_key_rejects_other_user_preferences_and_updates_last_used_at(
    user_history_client,
):
    client, session_factory = user_history_client
    other_user_id = uuid.uuid4()

    read_response = client.get(f"/users/{other_user_id}/preferences")
    write_response = client.put(
        f"/users/{other_user_id}/preferences/genre/comedy",
        json={"weight": 1.0, "source": "manual"},
    )

    assert read_response.status_code == 403
    assert write_response.status_code == 403
    assert read_response.json() == {
        "detail": "API key is not allowed to access this user."
    }
    assert write_response.json() == read_response.json()
    api_key_record = get_api_key_record(session_factory, DEFAULT_API_KEY)
    assert api_key_record.last_used_at is not None

    with session_factory() as session:
        other_user = session.get(User, other_user_id)
        other_preference = session.scalar(
            select(UserPreference).where(UserPreference.user_id == other_user_id)
        )

    assert other_user is None
    assert other_preference is None


def test_api_client_without_owner_user_id_rejects_user_preferences(
    user_history_client,
):
    _, session_factory = user_history_client
    user_id = uuid.uuid4()
    no_owner_key = "oct_prefnoowner_test-secret"
    add_api_key(
        session_factory,
        api_key=no_owner_key,
        scopes=[PREFERENCES_READ_SCOPE],
    )
    client = TestClient(app)

    response = client.get(
        f"/users/{user_id}/preferences",
        headers={"Authorization": f"Bearer {no_owner_key}"},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "API key is not allowed to access this user."}
    api_key_record = get_api_key_record(session_factory, no_owner_key)
    assert api_key_record.last_used_at is not None


def test_put_user_history_updates_owner_history(user_history_client):
    client, session_factory = user_history_client
    user_id = DEFAULT_OWNER_USER_ID
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
    user_id = DEFAULT_OWNER_USER_ID
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
    user_id = DEFAULT_OWNER_USER_ID
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
    user_id = DEFAULT_OWNER_USER_ID
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
    user_id = DEFAULT_OWNER_USER_ID
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
    user_id = DEFAULT_OWNER_USER_ID

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
    user_id = DEFAULT_OWNER_USER_ID
    add_catalog_movies(session_factory, "862")

    response = client.put(
        f"/users/{user_id}/ratings/862",
        json={"rating_value": 10.5, "source": "manual"},
    )

    assert response.status_code == 422
