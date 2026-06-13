from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import and_, desc, select
from sqlalchemy.orm import Session

from backend.app.db.models import (
    MovieCatalogEntry,
    User,
    UserMovieHistory,
    UserMovieRating,
)
from backend.app.db.session import get_db_session


HistoryStatus = Literal["watched", "planned", "dropped"]
UserDataSource = Literal["manual", "csv_import", "api", "system"]

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


@router.get("/{user_id}/history", response_model=UserHistoryResponse)
def get_user_history(
    user_id: uuid.UUID,
    status: HistoryStatus | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_db_session),
) -> UserHistoryResponse:
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


@router.put(
    "/{user_id}/history/{movie_id}",
    response_model=UserHistoryItem,
)
def put_user_history(
    user_id: uuid.UUID,
    movie_id: str,
    request: UserHistoryUpdateRequest,
    session: Session = Depends(get_db_session),
) -> UserHistoryItem:
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
    session: Session = Depends(get_db_session),
) -> UserRatingResponse:
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
