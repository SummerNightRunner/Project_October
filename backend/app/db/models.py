from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
import uuid

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, CITEXT, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


timestamp_with_timezone = sa.DateTime(timezone=True)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        sa.CheckConstraint(
            "status in ('active', 'disabled', 'deleted')",
            name="users_status_values",
        ),
        sa.Index(
            "users_email_key",
            "email",
            unique=True,
            postgresql_where=sa.text("email IS NOT NULL AND email <> ''"),
        ),
        sa.Index("users_status_idx", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=sa.text("gen_random_uuid()"),
    )
    email: Mapped[str | None] = mapped_column(CITEXT())
    display_name: Mapped[str | None] = mapped_column(sa.Text())
    status: Mapped[str] = mapped_column(sa.Text(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        timestamp_with_timezone,
        nullable=False,
        server_default=sa.func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        timestamp_with_timezone,
        nullable=False,
        server_default=sa.func.now(),
    )


class MovieCatalogEntry(Base):
    __tablename__ = "movie_catalog_entries"
    __table_args__ = (
        sa.Index("movie_catalog_entries_title_idx", "title_snapshot"),
    )

    catalog_movie_id: Mapped[str] = mapped_column(sa.Text(), primary_key=True)
    title_snapshot: Mapped[str | None] = mapped_column(sa.Text())
    release_date: Mapped[date | None] = mapped_column(sa.Date())
    source_catalog_version: Mapped[str | None] = mapped_column(sa.Text())
    created_at: Mapped[datetime] = mapped_column(
        timestamp_with_timezone,
        nullable=False,
        server_default=sa.func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        timestamp_with_timezone,
        nullable=False,
        server_default=sa.func.now(),
    )


class UserMovieHistory(Base):
    __tablename__ = "user_movie_history"
    __table_args__ = (
        sa.CheckConstraint(
            "status in ('watched', 'planned', 'dropped')",
            name="user_movie_history_status_values",
        ),
        sa.CheckConstraint(
            "source in ('manual', 'csv_import', 'api', 'system')",
            name="user_movie_history_source_values",
        ),
        sa.UniqueConstraint(
            "user_id",
            "catalog_movie_id",
            name="user_movie_history_user_movie_key",
        ),
        sa.Index("user_movie_history_user_status_idx", "user_id", "status"),
        sa.Index(
            "user_movie_history_user_watched_at_idx",
            "user_id",
            sa.desc("watched_at"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=sa.text("gen_random_uuid()"),
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    catalog_movie_id: Mapped[str] = mapped_column(
        sa.Text(),
        sa.ForeignKey("movie_catalog_entries.catalog_movie_id"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(sa.Text(), nullable=False)
    watched_at: Mapped[datetime | None] = mapped_column(timestamp_with_timezone)
    source: Mapped[str] = mapped_column(sa.Text(), nullable=False)
    notes: Mapped[str | None] = mapped_column(sa.Text())
    created_at: Mapped[datetime] = mapped_column(
        timestamp_with_timezone,
        nullable=False,
        server_default=sa.func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        timestamp_with_timezone,
        nullable=False,
        server_default=sa.func.now(),
    )


class UserMovieRating(Base):
    __tablename__ = "user_movie_ratings"
    __table_args__ = (
        sa.CheckConstraint(
            "rating_value >= 0 and rating_value <= 10",
            name="user_movie_ratings_rating_value_range",
        ),
        sa.CheckConstraint(
            "source in ('manual', 'csv_import', 'api', 'system')",
            name="user_movie_ratings_source_values",
        ),
        sa.UniqueConstraint(
            "user_id",
            "catalog_movie_id",
            name="user_movie_ratings_user_movie_key",
        ),
        sa.Index(
            "user_movie_ratings_user_rating_idx",
            "user_id",
            sa.desc("rating_value"),
        ),
        sa.Index("user_movie_ratings_movie_idx", "catalog_movie_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=sa.text("gen_random_uuid()"),
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    catalog_movie_id: Mapped[str] = mapped_column(
        sa.Text(),
        sa.ForeignKey("movie_catalog_entries.catalog_movie_id"),
        nullable=False,
    )
    rating_value: Mapped[Decimal] = mapped_column(sa.Numeric(3, 1), nullable=False)
    rated_at: Mapped[datetime | None] = mapped_column(timestamp_with_timezone)
    source: Mapped[str] = mapped_column(sa.Text(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        timestamp_with_timezone,
        nullable=False,
        server_default=sa.func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        timestamp_with_timezone,
        nullable=False,
        server_default=sa.func.now(),
    )


class UserPreference(Base):
    __tablename__ = "user_preferences"
    __table_args__ = (
        sa.CheckConstraint(
            "preference_type in ("
            "'genre', 'keyword', 'person', 'language', 'adult_content', "
            "'animation', 'free_text'"
            ")",
            name="user_preferences_preference_type_values",
        ),
        sa.CheckConstraint(
            "weight >= -10 and weight <= 10",
            name="user_preferences_weight_range",
        ),
        sa.CheckConstraint(
            "source in ('manual', 'csv_import', 'api', 'system')",
            name="user_preferences_source_values",
        ),
        sa.UniqueConstraint(
            "user_id",
            "preference_type",
            "preference_key",
            name="user_preferences_user_type_key_key",
        ),
        sa.Index("user_preferences_user_active_idx", "user_id", "is_active"),
        sa.Index(
            "user_preferences_type_key_idx",
            "preference_type",
            "preference_key",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=sa.text("gen_random_uuid()"),
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    preference_type: Mapped[str] = mapped_column(sa.Text(), nullable=False)
    preference_key: Mapped[str] = mapped_column(sa.Text(), nullable=False)
    weight: Mapped[Decimal] = mapped_column(sa.Numeric(4, 2), nullable=False)
    source: Mapped[str] = mapped_column(sa.Text(), nullable=False)
    is_active: Mapped[bool] = mapped_column(sa.Boolean(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        timestamp_with_timezone,
        nullable=False,
        server_default=sa.func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        timestamp_with_timezone,
        nullable=False,
        server_default=sa.func.now(),
    )


class ApiClient(Base):
    __tablename__ = "api_clients"
    __table_args__ = (
        sa.CheckConstraint(
            "status in ('active', 'disabled', 'deleted')",
            name="api_clients_status_values",
        ),
        sa.Index("api_clients_owner_idx", "owner_user_id"),
        sa.Index("api_clients_status_idx", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=sa.text("gen_random_uuid()"),
    )
    owner_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("users.id", ondelete="SET NULL"),
    )
    name: Mapped[str] = mapped_column(sa.Text(), nullable=False)
    contact_email: Mapped[str | None] = mapped_column(CITEXT())
    status: Mapped[str] = mapped_column(sa.Text(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        timestamp_with_timezone,
        nullable=False,
        server_default=sa.func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        timestamp_with_timezone,
        nullable=False,
        server_default=sa.func.now(),
    )


class ApiKey(Base):
    __tablename__ = "api_keys"
    __table_args__ = (
        sa.CheckConstraint(
            "status in ('active', 'revoked', 'expired')",
            name="api_keys_status_values",
        ),
        sa.UniqueConstraint("key_prefix", name="api_keys_key_prefix_key"),
        sa.Index("api_keys_client_status_idx", "api_client_id", "status"),
        sa.Index("api_keys_expires_at_idx", "expires_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=sa.text("gen_random_uuid()"),
    )
    api_client_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("api_clients.id", ondelete="CASCADE"),
        nullable=False,
    )
    key_prefix: Mapped[str] = mapped_column(sa.Text(), nullable=False)
    key_hash: Mapped[str] = mapped_column(sa.Text(), nullable=False)
    scopes: Mapped[list[str]] = mapped_column(
        ARRAY(sa.Text()),
        nullable=False,
        server_default=sa.text("ARRAY[]::text[]"),
    )
    status: Mapped[str] = mapped_column(sa.Text(), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(timestamp_with_timezone)
    last_used_at: Mapped[datetime | None] = mapped_column(timestamp_with_timezone)
    created_at: Mapped[datetime] = mapped_column(
        timestamp_with_timezone,
        nullable=False,
        server_default=sa.func.now(),
    )
    revoked_at: Mapped[datetime | None] = mapped_column(timestamp_with_timezone)


class UserEvent(Base):
    __tablename__ = "user_events"
    __table_args__ = (
        sa.Index("user_events_user_created_idx", "user_id", sa.desc("created_at")),
        sa.Index(
            "user_events_type_created_idx",
            "event_type",
            sa.desc("created_at"),
        ),
        sa.Index("user_events_request_id_idx", "request_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=sa.text("gen_random_uuid()"),
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("users.id", ondelete="SET NULL"),
    )
    api_client_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("api_clients.id", ondelete="SET NULL"),
    )
    event_type: Mapped[str] = mapped_column(sa.Text(), nullable=False)
    catalog_movie_id: Mapped[str | None] = mapped_column(
        sa.Text(),
        sa.ForeignKey("movie_catalog_entries.catalog_movie_id"),
    )
    payload: Mapped[dict[str, object]] = mapped_column(
        JSONB(),
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )
    request_id: Mapped[str | None] = mapped_column(sa.Text())
    created_at: Mapped[datetime] = mapped_column(
        timestamp_with_timezone,
        nullable=False,
        server_default=sa.func.now(),
    )
