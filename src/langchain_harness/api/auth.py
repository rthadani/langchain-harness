"""Request authentication: a pluggable chain of auth modes -> user dict.

The server accepts one or more modes, listed in order by ``[server]
auth_modes``. Each mode inspects the request and either claims it (it found a
credential of its own kind) or passes. The first claim wins, so the order is
the precedence.

Modes:

- ``trust``: ``X-User-Id: <id>`` believed as-is, for deployments behind an
  authenticating gateway.
- ``jwt``: ``Authorization: Bearer <JWT>`` verified against the issuer's JWKS
  (OIDC discovery). The user id is taken from the configured claim.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from fastapi import HTTPException, Request, status

from langchain_harness.config import KNOWN_AUTH_MODES, JwtConfig, ServerConfig

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


class TrustHeaderAuthenticator:
    """``X-User-Id`` taken at face value; the caller is already trusted."""

    name = "trust"

    async def resolve(self, request: Request) -> dict[str, Any] | None:
        raw = request.headers.get(USER_ID_HEADER)
        if raw is None:
            return None
        return {"user_id": sanitize_user_id(raw)}


class JwtAuthenticator:
    """Bearer JWT verified against OIDC discovery + JWKS."""

    name = "jwt"

    def __init__(self, config: JwtConfig) -> None:
        self.config = config
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
        return {"user_id": user_id}

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


def build_auth(config: ServerConfig) -> AuthChain:
    """Build the auth chain described by ``config.auth_modes``."""
    modes = tuple(config.auth_modes) or ("trust",)
    unknown = [mode for mode in modes if mode not in KNOWN_AUTH_MODES]
    if unknown:
        raise ValueError(
            f"Unknown auth mode {unknown[0]!r}; "
            f"known modes: {', '.join(KNOWN_AUTH_MODES)}"
        )

    builders = {
        "trust": lambda: TrustHeaderAuthenticator(),
        "jwt": lambda: JwtAuthenticator(config.jwt),
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
    return chain


async def get_current_user(request: Request) -> dict[str, Any]:
    """FastAPI dependency: resolve the request to a user dict.

    Raises 400 (missing credential under ``trust``), 401 (invalid token), or
    503 (identity provider unreachable).
    """
    return await request.app.state.auth.authenticate(request)
