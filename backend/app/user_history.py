from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import and_, desc, select
from sqlalchemy.orm import Session

from backend.app.api_key_auth import (
    ApiKeyPrincipal,
    HISTORY_READ_SCOPE,
    HISTORY_WRITE_SCOPE,
    PREFERENCES_READ_SCOPE,
    PREFERENCES_WRITE_SCOPE,
    RATINGS_WRITE_SCOPE,
    ensure_api_client_can_access_user,
    require_api_scope,
)
from backend.app.db.models import (
    MovieCatalogEntry,
    User,
    UserMovieHistory,
    UserMovieRating,
    UserPreference,
)
from backend.app.db.session import get_db_session


HistoryStatus = Literal["watched", "planned", "dropped"]
UserDataSource = Literal["manual", "csv_import", "api", "system"]
UserPreferenceType = Literal[
    "genre",
    "keyword",
    "person",
    "language",
    "adult_content",
    "animation",
    "free_text",
]

router = APIRouter(prefix="/users", tags=["users"])


class UserHistoryUpdateRequest(BaseModel):
    status: HistoryStatus
    watched_at: datetime | None = None
    source: UserDataSource = "manual"
    notes: str | None = Field(default=None, max_length=2000)

    @field_validator("notes")
    @classmethod
    def normalize_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized_value = value.strip()
        return normalized_value or None


class UserRatingUpdateRequest(BaseModel):
    rating_value: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("10"),
        max_digits=3,
        decimal_places=1,
    )
    rated_at: datetime | None = None
    source: UserDataSource = "manual"


class UserPreferenceUpdateRequest(BaseModel):
    weight: Decimal = Field(
        ...,
        ge=Decimal("-10.00"),
        le=Decimal("10.00"),
        max_digits=4,
        decimal_places=2,
    )
    source: UserDataSource = "manual"
    is_active: bool = True


class UserHistoryItem(BaseModel):
    movie_id: str
    status: HistoryStatus
    watched_at: datetime | None = None
    rating_value: float | None = None
    source: UserDataSource
    notes: str | None = None


class UserHistoryResponse(BaseModel):
    items: list[UserHistoryItem]


class UserRatingResponse(BaseModel):
    movie_id: str
    rating_value: float
    rated_at: datetime | None = None
    source: UserDataSource


class UserPreferenceItem(BaseModel):
    preference_type: UserPreferenceType
    preference_key: str
    weight: float
    source: UserDataSource
    is_active: bool
    created_at: datetime
    updated_at: datetime


class UserPreferencesResponse(BaseModel):
    items: list[UserPreferenceItem]


def utc_now() -> datetime:
    return datetime.now(UTC)


def decimal_to_float(value: Decimal | None) -> float | None:
    if value is None:
        return None
    return float(value)


def as_utc_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def ensure_movie_exists(session: Session, movie_id: str) -> MovieCatalogEntry:
    movie = session.get(MovieCatalogEntry, movie_id)
    if movie is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Movie with movie_id '{movie_id}' was not found in "
                "movie_catalog_entries."
            ),
        )
    return movie


def ensure_user_exists(session: Session, user_id: uuid.UUID) -> User:
    user = session.get(User, user_id)
    if user is not None:
        return user

    created_at = utc_now()
    user = User(
        id=user_id,
        status="active",
        created_at=created_at,
        updated_at=created_at,
    )
    session.add(user)
    session.flush()
    return user


def build_history_item(
    history_entry: UserMovieHistory,
    rating_value: Decimal | None,
) -> UserHistoryItem:
    return UserHistoryItem(
        movie_id=history_entry.catalog_movie_id,
        status=history_entry.status,
        watched_at=as_utc_datetime(history_entry.watched_at),
        rating_value=decimal_to_float(rating_value),
        source=history_entry.source,
        notes=history_entry.notes,
    )


def get_history_item(
    session: Session,
    user_id: uuid.UUID,
    movie_id: str,
) -> UserHistoryItem:
    query = (
        select(UserMovieHistory, UserMovieRating.rating_value)
        .outerjoin(
            UserMovieRating,
            and_(
                UserMovieRating.user_id == UserMovieHistory.user_id,
                UserMovieRating.catalog_movie_id == UserMovieHistory.catalog_movie_id,
            ),
        )
        .where(
            UserMovieHistory.user_id == user_id,
            UserMovieHistory.catalog_movie_id == movie_id,
        )
    )
    row = session.execute(query).one()
    return build_history_item(row[0], row[1])


def normalize_preference_key(preference_key: str) -> str:
    normalized_key = preference_key.strip()
    if not normalized_key:
        raise HTTPException(status_code=422, detail="preference_key must not be empty.")
    return normalized_key


def build_preference_item(preference: UserPreference) -> UserPreferenceItem:
    return UserPreferenceItem(
        preference_type=preference.preference_type,
        preference_key=preference.preference_key,
        weight=float(preference.weight),
        source=preference.source,
        is_active=preference.is_active,
        created_at=as_utc_datetime(preference.created_at),
        updated_at=as_utc_datetime(preference.updated_at),
    )


@router.get("/{user_id}/history", response_model=UserHistoryResponse)
def get_user_history(
    user_id: uuid.UUID,
    status: HistoryStatus | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    auth: ApiKeyPrincipal = Depends(require_api_scope(HISTORY_READ_SCOPE)),
    session: Session = Depends(get_db_session),
) -> UserHistoryResponse:
    ensure_api_client_can_access_user(auth, user_id)

    query = (
        select(UserMovieHistory, UserMovieRating.rating_value)
        .outerjoin(
            UserMovieRating,
            and_(
                UserMovieRating.user_id == UserMovieHistory.user_id,
                UserMovieRating.catalog_movie_id == UserMovieHistory.catalog_movie_id,
            ),
        )
        .where(UserMovieHistory.user_id == user_id)
        .order_by(
            UserMovieHistory.watched_at.is_(None),
            desc(UserMovieHistory.watched_at),
            desc(UserMovieHistory.updated_at),
        )
        .limit(limit)
    )

    if status is not None:
        query = query.where(UserMovieHistory.status == status)

    rows = session.execute(query).all()
    return UserHistoryResponse(
        items=[
            build_history_item(history_entry=row[0], rating_value=row[1])
            for row in rows
        ]
    )


@router.get("/{user_id}/preferences", response_model=UserPreferencesResponse)
def get_user_preferences(
    user_id: uuid.UUID,
    is_active: bool | None = Query(default=None),
    preference_type: UserPreferenceType | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    auth: ApiKeyPrincipal = Depends(require_api_scope(PREFERENCES_READ_SCOPE)),
    session: Session = Depends(get_db_session),
) -> UserPreferencesResponse:
    ensure_api_client_can_access_user(auth, user_id)

    query = (
        select(UserPreference)
        .where(UserPreference.user_id == user_id)
        .order_by(desc(UserPreference.updated_at), desc(UserPreference.created_at))
        .limit(limit)
    )

    if is_active is not None:
        query = query.where(UserPreference.is_active == is_active)

    if preference_type is not None:
        query = query.where(UserPreference.preference_type == preference_type)

    preferences = session.scalars(query).all()
    return UserPreferencesResponse(
        items=[build_preference_item(preference) for preference in preferences]
    )


@router.put(
    "/{user_id}/preferences/{preference_type}/{preference_key}",
    response_model=UserPreferenceItem,
)
def put_user_preference(
    user_id: uuid.UUID,
    preference_type: UserPreferenceType,
    preference_key: str,
    request: UserPreferenceUpdateRequest,
    auth: ApiKeyPrincipal = Depends(require_api_scope(PREFERENCES_WRITE_SCOPE)),
    session: Session = Depends(get_db_session),
) -> UserPreferenceItem:
    ensure_api_client_can_access_user(auth, user_id)
    ensure_user_exists(session, user_id)
    normalized_preference_key = normalize_preference_key(preference_key)
    updated_at = utc_now()

    preference = session.scalar(
        select(UserPreference).where(
            UserPreference.user_id == user_id,
            UserPreference.preference_type == preference_type,
            UserPreference.preference_key == normalized_preference_key,
        )
    )

    if preference is None:
        preference = UserPreference(
            user_id=user_id,
            preference_type=preference_type,
            preference_key=normalized_preference_key,
            weight=request.weight,
            source=request.source,
            is_active=request.is_active,
            created_at=updated_at,
            updated_at=updated_at,
        )
        session.add(preference)
    else:
        preference.weight = request.weight
        preference.source = request.source
        preference.is_active = request.is_active
        preference.updated_at = updated_at

    session.commit()
    return build_preference_item(preference)


@router.put(
    "/{user_id}/history/{movie_id}",
    response_model=UserHistoryItem,
)
def put_user_history(
    user_id: uuid.UUID,
    movie_id: str,
    request: UserHistoryUpdateRequest,
    auth: ApiKeyPrincipal = Depends(require_api_scope(HISTORY_WRITE_SCOPE)),
    session: Session = Depends(get_db_session),
) -> UserHistoryItem:
    ensure_api_client_can_access_user(auth, user_id)
    ensure_movie_exists(session, movie_id)
    ensure_user_exists(session, user_id)
    updated_at = utc_now()

    history_entry = session.scalar(
        select(UserMovieHistory).where(
            UserMovieHistory.user_id == user_id,
            UserMovieHistory.catalog_movie_id == movie_id,
        )
    )

    if history_entry is None:
        history_entry = UserMovieHistory(
            user_id=user_id,
            catalog_movie_id=movie_id,
            status=request.status,
            watched_at=request.watched_at,
            source=request.source,
            notes=request.notes,
            created_at=updated_at,
            updated_at=updated_at,
        )
        session.add(history_entry)
    else:
        history_entry.status = request.status
        history_entry.watched_at = request.watched_at
        history_entry.source = request.source
        history_entry.notes = request.notes
        history_entry.updated_at = updated_at

    session.commit()
    return get_history_item(session=session, user_id=user_id, movie_id=movie_id)


@router.put(
    "/{user_id}/ratings/{movie_id}",
    response_model=UserRatingResponse,
)
def put_user_rating(
    user_id: uuid.UUID,
    movie_id: str,
    request: UserRatingUpdateRequest,
    auth: ApiKeyPrincipal = Depends(require_api_scope(RATINGS_WRITE_SCOPE)),
    session: Session = Depends(get_db_session),
) -> UserRatingResponse:
    ensure_api_client_can_access_user(auth, user_id)
    ensure_movie_exists(session, movie_id)
    ensure_user_exists(session, user_id)
    updated_at = utc_now()

    rating_entry = session.scalar(
        select(UserMovieRating).where(
            UserMovieRating.user_id == user_id,
            UserMovieRating.catalog_movie_id == movie_id,
        )
    )

    if rating_entry is None:
        rating_entry = UserMovieRating(
            user_id=user_id,
            catalog_movie_id=movie_id,
            rating_value=request.rating_value,
            rated_at=request.rated_at,
            source=request.source,
            created_at=updated_at,
            updated_at=updated_at,
        )
        session.add(rating_entry)
    else:
        rating_entry.rating_value = request.rating_value
        rating_entry.rated_at = request.rated_at
        rating_entry.source = request.source
        rating_entry.updated_at = updated_at

    session.commit()
    return UserRatingResponse(
        movie_id=rating_entry.catalog_movie_id,
        rating_value=float(rating_entry.rating_value),
        rated_at=as_utc_datetime(rating_entry.rated_at),
        source=rating_entry.source,
    )
