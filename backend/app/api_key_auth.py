from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import hmac
import secrets
import uuid

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import ApiClient, ApiKey
from backend.app.db.session import get_db_session


HISTORY_READ_SCOPE = "history:read"
HISTORY_WRITE_SCOPE = "history:write"
RATINGS_WRITE_SCOPE = "ratings:write"
PREFERENCES_READ_SCOPE = "preferences:read"
PREFERENCES_WRITE_SCOPE = "preferences:write"
RECOMMENDATIONS_READ_SCOPE = "recommendations:read"

SUPPORTED_API_KEY_SCOPES = frozenset(
    {
        HISTORY_READ_SCOPE,
        HISTORY_WRITE_SCOPE,
        RATINGS_WRITE_SCOPE,
        PREFERENCES_READ_SCOPE,
        PREFERENCES_WRITE_SCOPE,
        RECOMMENDATIONS_READ_SCOPE,
    }
)

API_KEY_HASH_ALGORITHM = "sha256"
API_KEY_PREFIX_NAMESPACE = "oct"


@dataclass(frozen=True)
class ApiKeyPrincipal:
    api_client_id: uuid.UUID
    api_key_id: uuid.UUID
    key_prefix: str
    owner_user_id: uuid.UUID | None
    scopes: frozenset[str]


def utc_now() -> datetime:
    return datetime.now(UTC)


def create_api_key_hash(api_key: str) -> str:
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return f"{API_KEY_HASH_ALGORITHM}:{digest}"


def verify_api_key_hash(api_key: str, key_hash: str) -> bool:
    expected_hash = create_api_key_hash(api_key)
    return hmac.compare_digest(expected_hash, key_hash)


def create_api_key(prefix: str | None = None) -> tuple[str, str]:
    resolved_prefix = prefix or secrets.token_hex(4)
    if not resolved_prefix or "_" in resolved_prefix:
        raise ValueError("API key prefix must be non-empty and must not contain '_'.")

    key_prefix = f"{API_KEY_PREFIX_NAMESPACE}_{resolved_prefix}"
    secret = secrets.token_urlsafe(32)
    return f"{key_prefix}_{secret}", key_prefix


def extract_api_key_prefix(api_key: str) -> str | None:
    parts = api_key.split("_", 2)
    if len(parts) != 3:
        return None

    namespace, prefix, secret = parts
    if namespace != API_KEY_PREFIX_NAMESPACE or not prefix or not secret:
        return None

    return f"{namespace}_{prefix}"


def unauthorized(detail: str = "Invalid API key.") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def forbidden(detail: str = "API key does not have the required scope.") -> HTTPException:
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


def forbidden_user_access() -> HTTPException:
    return forbidden("API key is not allowed to access this user.")


def extract_bearer_token(authorization: str | None) -> str:
    if authorization is None:
        raise unauthorized("Missing API key.")

    scheme, separator, credentials = authorization.strip().partition(" ")
    if (
        not separator
        or scheme.casefold() != "bearer"
        or not credentials
        or " " in credentials.strip()
    ):
        raise unauthorized("Invalid Authorization header.")

    return credentials.strip()


def as_utc_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def verify_api_key(
    *,
    session: Session,
    api_key: str,
    required_scope: str,
) -> ApiKeyPrincipal:
    if required_scope not in SUPPORTED_API_KEY_SCOPES:
        raise RuntimeError(f"Unsupported API key scope: {required_scope}")

    key_prefix = extract_api_key_prefix(api_key)
    if key_prefix is None:
        raise unauthorized()

    api_key_record = session.scalar(
        select(ApiKey).where(ApiKey.key_prefix == key_prefix)
    )
    if api_key_record is None or not verify_api_key_hash(
        api_key=api_key,
        key_hash=api_key_record.key_hash,
    ):
        raise unauthorized()

    api_client = session.get(ApiClient, api_key_record.api_client_id)
    if (
        api_key_record.status != "active"
        or api_client is None
        or api_client.status != "active"
    ):
        raise unauthorized()

    expires_at = as_utc_datetime(api_key_record.expires_at)
    if expires_at is not None and expires_at <= utc_now():
        raise unauthorized("API key is expired.")

    scopes = frozenset(api_key_record.scopes)
    if required_scope not in scopes:
        raise forbidden()

    api_key_record.last_used_at = utc_now()
    session.commit()

    return ApiKeyPrincipal(
        api_client_id=api_key_record.api_client_id,
        api_key_id=api_key_record.id,
        key_prefix=api_key_record.key_prefix,
        owner_user_id=api_client.owner_user_id,
        scopes=scopes,
    )


def ensure_api_client_can_access_user(
    principal: ApiKeyPrincipal,
    user_id: uuid.UUID,
) -> None:
    if principal.owner_user_id is None or principal.owner_user_id != user_id:
        raise forbidden_user_access()


def require_api_scope(required_scope: str):
    def dependency(
        authorization: str | None = Header(default=None, alias="Authorization"),
        session: Session = Depends(get_db_session),
    ) -> ApiKeyPrincipal:
        api_key = extract_bearer_token(authorization)
        return verify_api_key(
            session=session,
            api_key=api_key,
            required_scope=required_scope,
        )

    return dependency
