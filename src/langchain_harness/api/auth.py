"""Request authentication: a pluggable chain of auth modes -> user dict.

The server accepts one or more modes, listed in order by ``[server]
auth_modes``. Each mode inspects the request and either claims it (it found a
credential of its own kind) or passes. The first claim wins, so the order is
the precedence.

Modes:

- ``token``: ``Authorization: Bearer harn_...`` looked up in the ``users``
  table. Users are created ahead of time with ``harness users add``.
- ``jwt``: ``Authorization: Bearer <JWT>`` verified against the issuer's JWKS
  (OIDC discovery). Users are created on first valid token.
- ``trust``: ``X-User-Id: <id>`` believed as-is, for tests and deployments
  behind an authenticating gateway. Users are created on first request.

The token and jwt modes split on credential shape (a JWT has two dots), so
listing them in either order behaves the same. ``trust`` reads a different
header and never collides with either.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from fastapi import HTTPException, Request, status

from langchain_harness.config import KNOWN_AUTH_MODES, JwtConfig, ServerConfig
from langchain_harness.serverdb import UserIdConflict
from langchain_harness.store import Store

logger = logging.getLogger(__name__)

#: Header read by the ``trust`` mode.
USER_ID_HEADER = "X-User-Id"

# The user id becomes a directory name under ``.harness/users/``, so the
# character set is deliberately narrow: no separators, no leading dot or dash,
# nothing that could climb out of the users root.
_USER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9@._+-]{0,127}$")


def sanitize_user_id(raw: Any) -> str:
    """Return a filesystem-safe user id, or raise 400.

    Rejecting here (rather than at the workspace layer) keeps a hostile
    ``X-User-Id`` or JWT claim from reaching ``provision_user_workspace``.
    """
    candidate = str(raw or "").strip()
    if not _USER_ID_RE.match(candidate):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid user id: use 1-128 characters from A-Z a-z 0-9 @ . _ + - "
                "starting with a letter or digit."
            ),
        )
    return candidate


class Authenticator(Protocol):
    """One auth mode: claim a request or pass on it."""

    name: str

    async def resolve(self, request: Request) -> dict[str, Any] | None:
        """Return a user dict, ``None`` if this mode has no credential for the
        request, or raise ``HTTPException`` when its credential is invalid."""
        ...


def _bearer_token(request: Request) -> str | None:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    return header[len("Bearer "):].strip() or None


def _looks_like_jwt(token: str) -> bool:
    """A JWS compact serialization is exactly three dot-separated segments."""
    return token.count(".") == 2


async def _provisioned_user(
    store: Store,
    user_id: str,
    source: str,
    *,
    max_auto_provisioned_users: int = 500,
) -> dict[str, Any]:
    """Resolve (creating if new) the user for an externally authenticated id."""
    existing = await store.get_user_by_username(user_id)
    if existing is not None:
        effective_source = existing.get("auth_source") or "token"
        if effective_source != source:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"User id '{user_id}' belongs to a different authentication source",
            )
        if not existing["is_active"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="User is revoked"
            )
        return existing

    if max_auto_provisioned_users > 0:
        count = await store.count_provisioned_users()
        if count >= max_auto_provisioned_users:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Auto-provisioned user cap reached ({count}/"
                    f"{max_auto_provisioned_users}). Raise [server] "
                    "max_auto_provisioned_users or set "
                    "HARNESS_SERVER_MAX_AUTO_PROVISIONED_USERS."
                ),
            )

    try:
        return await store.get_or_create_user(user_id, source=source)
    except UserIdConflict:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User id '{user_id}' belongs to a different authentication source",
        ) from None


class TokenAuthenticator:
    """``Authorization: Bearer harn_...`` against the users table."""

    name = "token"

    def __init__(self, store: Store) -> None:
        self.store = store

    async def resolve(self, request: Request) -> dict[str, Any] | None:
        token = _bearer_token(request)
        if token is None or _looks_like_jwt(token):
            return None

        token_hash = hashlib.sha256(token.encode()).hexdigest()
        user = await self.store.get_user_by_token_hash(token_hash)
        if user is None or not user["is_active"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or revoked token",
            )
        return user


class TrustHeaderAuthenticator:
    """``X-User-Id`` taken at face value; the caller is already trusted."""

    name = "trust"

    def __init__(
        self, store: Store, *, max_auto_provisioned_users: int = 500
    ) -> None:
        self.store = store
        self.max_auto_provisioned_users = max_auto_provisioned_users

    async def resolve(self, request: Request) -> dict[str, Any] | None:
        raw = request.headers.get(USER_ID_HEADER)
        if raw is None:
            return None
        return await _provisioned_user(
            self.store,
            sanitize_user_id(raw),
            "trust",
            max_auto_provisioned_users=self.max_auto_provisioned_users,
        )


class JwtAuthenticator:
    """Bearer JWT verified against OIDC discovery + JWKS."""

    name = "jwt"

    def __init__(
        self,
        store: Store,
        config: JwtConfig,
        *,
        max_auto_provisioned_users: int = 500,
    ) -> None:
        self.store = store
        self.config = config
        self.max_auto_provisioned_users = max_auto_provisioned_users
        self._jwk_client: Any = None
        # Serializes discovery so a burst of first requests does not stampede
        # the identity provider.
        self._discovery_lock = asyncio.Lock()

    async def resolve(self, request: Request) -> dict[str, Any] | None:
        token = _bearer_token(request)
        if token is None or not _looks_like_jwt(token):
            return None

        claims = await self._decode(token)
        user_id = sanitize_user_id(claims.get(self.config.user_claim))
        return await _provisioned_user(
            self.store,
            user_id,
            "jwt",
            max_auto_provisioned_users=self.max_auto_provisioned_users,
        )

    async def _decode(self, token: str) -> dict[str, Any]:
        jwt_module = _import_pyjwt()
        client = await self._get_jwk_client(jwt_module)
        try:
            # PyJWKClient fetches JWKS over the network on cache miss, so keep
            # it off the event loop.
            signing_key = await asyncio.to_thread(
                client.get_signing_key_from_jwt, token
            )
            return jwt_module.decode(
                token,
                signing_key.key,
                algorithms=list(self.config.algorithms),
                audience=self.config.audience,
                issuer=self.config.issuer,
                leeway=self.config.clock_skew_seconds,
                options={
                    "require": ["exp"],
                },
            )
        except jwt_module.PyJWKClientConnectionError as exc:
            logger.warning("JWKS endpoint unreachable: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Token verification is temporarily unavailable",
            ) from exc
        except jwt_module.PyJWTError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {exc}",
            ) from exc

    async def _get_jwk_client(self, jwt_module: Any) -> Any:
        if self._jwk_client is not None:
            return self._jwk_client

        async with self._discovery_lock:
            if self._jwk_client is None:
                jwks_url = await self._discover(jwt_module)
                self._jwk_client = jwt_module.PyJWKClient(
                    jwks_url, cache_keys=True, lifespan=3600
                )
                logger.info("JWT auth using JWKS at %s", jwks_url)
            return self._jwk_client

    async def _discover(self, jwt_module: Any) -> str:
        """Return the JWKS URL, from config or OIDC discovery."""
        if self.config.jwks_url:
            return self.config.jwks_url
        if not self.config.issuer:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="JWT auth is not configured (no issuer)",
            )

        import httpx

        url = self.config.issuer.rstrip("/") + "/.well-known/openid-configuration"
        try:
            async with httpx.AsyncClient(
                timeout=self.config.discovery_timeout_seconds
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                document = response.json()
        except Exception as exc:  # noqa: BLE001 - network/JSON/HTTP all mean the same
            logger.warning("OIDC discovery failed for %s: %s", url, exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Token verification is temporarily unavailable",
            ) from exc

        jwks_url = document.get("jwks_uri") if isinstance(document, dict) else None
        if not jwks_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Issuer discovery document has no jwks_uri",
            )
        return str(jwks_url)


def _import_pyjwt() -> Any:
    try:
        import jwt
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise RuntimeError(
            "The 'jwt' auth mode needs PyJWT. Install "
            "'langchain-harness[jwt]' or 'PyJWT[crypto]'."
        ) from exc
    return jwt


@dataclass
class AuthChain:
    """Auth modes in precedence order."""

    authenticators: list[Authenticator] = field(default_factory=list)

    @property
    def modes(self) -> tuple[str, ...]:
        return tuple(auth.name for auth in self.authenticators)

    async def authenticate(self, request: Request) -> dict[str, Any]:
        for authenticator in self.authenticators:
            user = await authenticator.resolve(request)
            if user is not None:
                return user

        # A trust-only deployment means the caller forgot the header; say so
        # instead of a generic 401. With several modes there is no single
        # missing credential to name, so 401 is right.
        if self.modes == ("trust",):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing {USER_ID_HEADER} header",
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid credentials",
        )


def build_auth(config: ServerConfig, store: Store) -> AuthChain:
    """Build the auth chain described by ``config.auth_modes``."""
    modes = tuple(config.auth_modes) or ("token",)
    unknown = [mode for mode in modes if mode not in KNOWN_AUTH_MODES]
    if unknown:
        raise ValueError(
            f"Unknown auth mode {unknown[0]!r}; "
            f"known modes: {', '.join(KNOWN_AUTH_MODES)}"
        )

    builders = {
        "token": lambda: TokenAuthenticator(store),
        "trust": lambda: TrustHeaderAuthenticator(
            store, max_auto_provisioned_users=config.max_auto_provisioned_users
        ),
        "jwt": lambda: JwtAuthenticator(
            store, config.jwt, max_auto_provisioned_users=config.max_auto_provisioned_users
        ),
    }
    chain = AuthChain([builders[mode]() for mode in modes])

    if "trust" in chain.modes:
        logger.warning(
            "Auth mode 'trust' is enabled: any caller that can reach this port "
            "can claim any user id via %s, read that user's workspace and "
            "sessions, and allocate new workspaces without limit. Put this "
            "behind an authenticating gateway, or use 'jwt'.",
            USER_ID_HEADER,
        )
    if "jwt" in chain.modes and not config.jwt.audience:
        logger.warning(
            "JWT auth has no audience configured; a token minted by the same "
            "issuer for a different application is accepted here. Set "
            "[server.jwt] audience to the value this API is registered under."
        )
    if "token" not in chain.modes:
        logger.info(
            "Auth mode 'token' is disabled; `harness users add` tokens will not "
            "authenticate. Revocation via `harness users revoke` still applies."
        )
    return chain


async def get_current_user(request: Request) -> dict[str, Any]:
    """FastAPI dependency: resolve the request to a user dict.

    Raises 400 (missing credential under ``trust``), 401 (invalid/revoked), or
    503 (identity provider unreachable).
    """
    return await request.app.state.auth.authenticate(request)
