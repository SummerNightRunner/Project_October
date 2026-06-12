from dataclasses import dataclass
from functools import lru_cache
import os


DATABASE_URL_ENV = "PROJECT_OCTOBER_DATABASE_URL"
DATABASE_ECHO_ENV = "PROJECT_OCTOBER_DATABASE_ECHO"


@dataclass(frozen=True)
class DatabaseSettings:
    database_url: str | None
    echo: bool = False
    pool_pre_ping: bool = True


def parse_bool_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().casefold() in {"1", "true", "yes", "on"}


@lru_cache
def get_database_settings() -> DatabaseSettings:
    raw_database_url = os.environ.get(DATABASE_URL_ENV)
    database_url = raw_database_url.strip() if raw_database_url else None
    return DatabaseSettings(
        database_url=database_url or None,
        echo=parse_bool_env(os.environ.get(DATABASE_ECHO_ENV)),
    )


def require_database_url(settings: DatabaseSettings | None = None) -> str:
    resolved_settings = settings or get_database_settings()
    if not resolved_settings.database_url:
        raise RuntimeError(f"{DATABASE_URL_ENV} is not set.")
    return resolved_settings.database_url
