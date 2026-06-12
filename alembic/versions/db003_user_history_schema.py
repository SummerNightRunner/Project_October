"""create user history schema

Revision ID: db003_user_history_schema
Revises:
Create Date: 2026-06-13 00:00:00.000000
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "db003_user_history_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "citext"')

    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("email", postgresql.CITEXT(), nullable=True),
        sa.Column("display_name", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status in ('active', 'disabled', 'deleted')",
            name="users_status_values",
        ),
        sa.PrimaryKeyConstraint("id", name="users_pkey"),
    )
    op.create_index(
        "users_email_key",
        "users",
        ["email"],
        unique=True,
        postgresql_where=sa.text("email IS NOT NULL AND email <> ''"),
    )
    op.create_index("users_status_idx", "users", ["status"])

    op.create_table(
        "movie_catalog_entries",
        sa.Column("catalog_movie_id", sa.Text(), nullable=False),
        sa.Column("title_snapshot", sa.Text(), nullable=True),
        sa.Column("release_date", sa.Date(), nullable=True),
        sa.Column("source_catalog_version", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint(
            "catalog_movie_id",
            name="movie_catalog_entries_pkey",
        ),
    )
    op.create_index(
        "movie_catalog_entries_title_idx",
        "movie_catalog_entries",
        ["title_snapshot"],
    )

    op.create_table(
        "api_clients",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("owner_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("contact_email", postgresql.CITEXT(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status in ('active', 'disabled', 'deleted')",
            name="api_clients_status_values",
        ),
        sa.ForeignKeyConstraint(
            ["owner_user_id"],
            ["users.id"],
            name="api_clients_owner_user_id_fkey",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="api_clients_pkey"),
    )
    op.create_index("api_clients_owner_idx", "api_clients", ["owner_user_id"])
    op.create_index("api_clients_status_idx", "api_clients", ["status"])

    op.create_table(
        "api_keys",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("api_client_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("key_prefix", sa.Text(), nullable=False),
        sa.Column("key_hash", sa.Text(), nullable=False),
        sa.Column(
            "scopes",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("ARRAY[]::text[]"),
            nullable=False,
        ),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status in ('active', 'revoked', 'expired')",
            name="api_keys_status_values",
        ),
        sa.ForeignKeyConstraint(
            ["api_client_id"],
            ["api_clients.id"],
            name="api_keys_api_client_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="api_keys_pkey"),
        sa.UniqueConstraint("key_prefix", name="api_keys_key_prefix_key"),
    )
    op.create_index(
        "api_keys_client_status_idx",
        "api_keys",
        ["api_client_id", "status"],
    )
    op.create_index("api_keys_expires_at_idx", "api_keys", ["expires_at"])

    op.create_table(
        "user_movie_history",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("catalog_movie_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("watched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status in ('watched', 'planned', 'dropped')",
            name="user_movie_history_status_values",
        ),
        sa.CheckConstraint(
            "source in ('manual', 'csv_import', 'api', 'system')",
            name="user_movie_history_source_values",
        ),
        sa.ForeignKeyConstraint(
            ["catalog_movie_id"],
            ["movie_catalog_entries.catalog_movie_id"],
            name="user_movie_history_catalog_movie_id_fkey",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="user_movie_history_user_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="user_movie_history_pkey"),
        sa.UniqueConstraint(
            "user_id",
            "catalog_movie_id",
            name="user_movie_history_user_movie_key",
        ),
    )
    op.create_index(
        "user_movie_history_user_status_idx",
        "user_movie_history",
        ["user_id", "status"],
    )
    op.create_index(
        "user_movie_history_user_watched_at_idx",
        "user_movie_history",
        ["user_id", sa.text("watched_at DESC")],
    )

    op.create_table(
        "user_movie_ratings",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("catalog_movie_id", sa.Text(), nullable=False),
        sa.Column("rating_value", sa.Numeric(3, 1), nullable=False),
        sa.Column("rated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "rating_value >= 0 and rating_value <= 10",
            name="user_movie_ratings_rating_value_range",
        ),
        sa.CheckConstraint(
            "source in ('manual', 'csv_import', 'api', 'system')",
            name="user_movie_ratings_source_values",
        ),
        sa.ForeignKeyConstraint(
            ["catalog_movie_id"],
            ["movie_catalog_entries.catalog_movie_id"],
            name="user_movie_ratings_catalog_movie_id_fkey",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="user_movie_ratings_user_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="user_movie_ratings_pkey"),
        sa.UniqueConstraint(
            "user_id",
            "catalog_movie_id",
            name="user_movie_ratings_user_movie_key",
        ),
    )
    op.create_index(
        "user_movie_ratings_user_rating_idx",
        "user_movie_ratings",
        ["user_id", sa.text("rating_value DESC")],
    )
    op.create_index(
        "user_movie_ratings_movie_idx",
        "user_movie_ratings",
        ["catalog_movie_id"],
    )

    op.create_table(
        "user_preferences",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("preference_type", sa.Text(), nullable=False),
        sa.Column("preference_key", sa.Text(), nullable=False),
        sa.Column("weight", sa.Numeric(4, 2), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
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
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="user_preferences_user_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="user_preferences_pkey"),
        sa.UniqueConstraint(
            "user_id",
            "preference_type",
            "preference_key",
            name="user_preferences_user_type_key_key",
        ),
    )
    op.create_index(
        "user_preferences_user_active_idx",
        "user_preferences",
        ["user_id", "is_active"],
    )
    op.create_index(
        "user_preferences_type_key_idx",
        "user_preferences",
        ["preference_type", "preference_key"],
    )

    op.create_table(
        "user_events",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("api_client_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("catalog_movie_id", sa.Text(), nullable=True),
        sa.Column(
            "payload",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("request_id", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["api_client_id"],
            ["api_clients.id"],
            name="user_events_api_client_id_fkey",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["catalog_movie_id"],
            ["movie_catalog_entries.catalog_movie_id"],
            name="user_events_catalog_movie_id_fkey",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="user_events_user_id_fkey",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="user_events_pkey"),
    )
    op.create_index(
        "user_events_user_created_idx",
        "user_events",
        ["user_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "user_events_type_created_idx",
        "user_events",
        ["event_type", sa.text("created_at DESC")],
    )
    op.create_index("user_events_request_id_idx", "user_events", ["request_id"])


def downgrade() -> None:
    op.drop_index("user_events_request_id_idx", table_name="user_events")
    op.drop_index("user_events_type_created_idx", table_name="user_events")
    op.drop_index("user_events_user_created_idx", table_name="user_events")
    op.drop_table("user_events")

    op.drop_index("user_preferences_type_key_idx", table_name="user_preferences")
    op.drop_index("user_preferences_user_active_idx", table_name="user_preferences")
    op.drop_table("user_preferences")

    op.drop_index("user_movie_ratings_movie_idx", table_name="user_movie_ratings")
    op.drop_index(
        "user_movie_ratings_user_rating_idx",
        table_name="user_movie_ratings",
    )
    op.drop_table("user_movie_ratings")

    op.drop_index(
        "user_movie_history_user_watched_at_idx",
        table_name="user_movie_history",
    )
    op.drop_index(
        "user_movie_history_user_status_idx",
        table_name="user_movie_history",
    )
    op.drop_table("user_movie_history")

    op.drop_index("api_keys_expires_at_idx", table_name="api_keys")
    op.drop_index("api_keys_client_status_idx", table_name="api_keys")
    op.drop_table("api_keys")

    op.drop_index("api_clients_status_idx", table_name="api_clients")
    op.drop_index("api_clients_owner_idx", table_name="api_clients")
    op.drop_table("api_clients")

    op.drop_index(
        "movie_catalog_entries_title_idx",
        table_name="movie_catalog_entries",
    )
    op.drop_table("movie_catalog_entries")

    op.drop_index("users_status_idx", table_name="users")
    op.drop_index("users_email_key", table_name="users")
    op.drop_table("users")
