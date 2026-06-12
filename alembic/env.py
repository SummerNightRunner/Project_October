from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from backend.app.db.base import Base
from backend.app.db.config import require_database_url
from backend.app.db import models  # noqa: F401


target_metadata = Base.metadata


def get_alembic_config():
    try:
        return context.config
    except Exception:
        return None


config = get_alembic_config()
if config is not None and config.config_file_name is not None:
    fileConfig(config.config_file_name)


def get_database_url() -> str:
    return require_database_url()


def run_migrations_offline() -> None:
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    if config is None:
        raise RuntimeError("Alembic config is not available.")

    config.set_main_option("sqlalchemy.url", get_database_url())
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


def should_run_migrations() -> bool:
    return config is not None


if should_run_migrations():
    if context.is_offline_mode():
        run_migrations_offline()
    else:
        run_migrations_online()
