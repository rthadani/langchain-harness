"""Tests for pluggable auth modes: token, trust, and jwt."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from langchain_harness.api.app import create_app
from langchain_harness.api.auth import (
    AuthChain,
    JwtAuthenticator,
    sanitize_user_id,
)
from langchain_harness.config import Config, JwtConfig, ServerConfig, resolve_config
from langchain_harness.serverdb import UserIdConflict, add_user, init_server_db, revoke_user, server_db_path
from langchain_harness.store import SqliteStore


class _FakeAgent:
    """Minimal fake agent for auth tests that only hit /v1/sessions."""

    def __init__(self) -> None:
        self.stream_calls: list[Any] = []

    async def astream_events(self, input_data: Any, config: Any, *, version: str):
        yield {"event": "on_chat_model_stream", "data": {"chunk": _Chunk("ok")}}

    async def aget_state(self, config: Any):
        return type("State", (), {"values": {"messages": []}})()


class _FakeAgentCache:
    def __init__(self, agent: Any | None = None) -> None:
        self._agent = agent or _FakeAgent()
        self.shared_model = None

    async def get_agent(self, user_id: str) -> Any:
        return self._agent

    async def close_all(self) -> None:
        pass


class _Chunk:
    def __init__(self, content: str) -> None:
        self.content = content


def _server_config(auth_modes: tuple[str, ...] = ("token",), **jwt_kwargs: Any) -> ServerConfig:
    return ServerConfig(auth_modes=auth_modes, jwt=JwtConfig(**jwt_kwargs))


async def _seed_user(tmp_path: str, username: str) -> str:
    """Add a token-backed user to server.db and return the plaintext token."""
    import aiosqlite

    db_path = server_db_path(tmp_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(str(db_path)) as conn:
        await init_server_db(conn)
        return await add_user(conn, username)


@asynccontextmanager
async def _app_client(tmp_path: Any, monkeypatch: Any, *, config: Config):
    """Create an app with a fake agent cache and yield its HTTP client."""
    monkeypatch.chdir(tmp_path)

    db_path = server_db_path(tmp_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    import aiosqlite

    async with aiosqlite.connect(str(db_path)) as conn:
        await init_server_db(conn)

    app = create_app(tmp_path, config)
    cache = _FakeAgentCache()

    async with app.router.lifespan_context(app):
        app.state.get_agent = cache.get_agent
        app.state.agent_cache = cache
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield {"app": app, "client": client, "tmp_path": tmp_path}


# --- trust-only mode ---


@pytest.mark.asyncio
async def test_trust_missing_header_returns_400(tmp_path, monkeypatch) -> None:
    config = Config(
        provider="anthropic",
        model="test-model",
        server=_server_config(("trust",)),
    )
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        resp = await env["client"].get("/v1/sessions")
        assert resp.status_code == 400
        assert "X-User-Id" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_trust_valid_header_creates_user(tmp_path, monkeypatch) -> None:
    config = Config(
        provider="anthropic",
        model="test-model",
        server=_server_config(("trust",)),
    )
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        resp = await env["client"].get(
            "/v1/sessions", headers={"X-User-Id": "alice"}
        )
        assert resp.status_code == 200

        store = SqliteStore(tmp_path)
        await store.initialize()
        users = await store.list_users()
        await store.close()

        assert len(users) == 1
        assert users[0]["username"] == "alice"
        assert users[0]["auth_source"] == "trust"


@pytest.mark.asyncio
async def test_trust_second_request_reuses_user_id(tmp_path, monkeypatch) -> None:
    config = Config(
        provider="anthropic",
        model="test-model",
        server=_server_config(("trust",)),
    )
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        resp = await env["client"].get(
            "/v1/sessions", headers={"X-User-Id": "alice"}
        )
        assert resp.status_code == 200

        store = SqliteStore(tmp_path)
        await store.initialize()
        first = await store.get_or_create_user("alice", source="trust")
        resp2 = await env["client"].get(
            "/v1/sessions", headers={"X-User-Id": "alice"}
        )
        rows = await store.list_users()
        await store.close()

        assert resp2.status_code == 200
        assert len(rows) == 1
        assert rows[0]["user_id"] == first["user_id"]


@pytest.mark.asyncio
async def test_trust_invalid_user_ids_return_400(tmp_path, monkeypatch) -> None:
    config = Config(
        provider="anthropic",
        model="test-model",
        server=_server_config(("trust",)),
    )
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        invalid = ["../etc", "a/b", "", "x" * 200]
        for value in invalid:
            resp = await env["client"].get(
                "/v1/sessions", headers={"X-User-Id": value}
            )
            assert resp.status_code == 400, value


@pytest.mark.asyncio
async def test_trust_revoked_user_returns_401(tmp_path, monkeypatch) -> None:
    config = Config(
        provider="anthropic",
        model="test-model",
        server=_server_config(("trust",)),
    )
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        # Auto-provision alice via trust, then revoke her.
        resp = await env["client"].get(
            "/v1/sessions", headers={"X-User-Id": "alice"}
        )
        assert resp.status_code == 200

        import aiosqlite

        async with aiosqlite.connect(str(server_db_path(tmp_path))) as conn:
            await revoke_user(conn, "alice")

        resp = await env["client"].get(
            "/v1/sessions", headers={"X-User-Id": "alice"}
        )
        assert resp.status_code == 401
        assert "revoked" in resp.json()["detail"].lower()


# --- token mode unchanged ---


@pytest.mark.asyncio
async def test_token_valid_bearer_returns_200(tmp_path, monkeypatch) -> None:
    token = await _seed_user(tmp_path, "alice")
    config = Config(provider="anthropic", model="test-model")
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        resp = await env["client"].get(
            "/v1/sessions", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_token_invalid_bearer_returns_401(tmp_path, monkeypatch) -> None:
    await _seed_user(tmp_path, "alice")
    config = Config(provider="anthropic", model="test-model")
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        resp = await env["client"].get(
            "/v1/sessions", headers={"Authorization": "Bearer harn_invalid"}
        )
        assert resp.status_code == 401


# --- chain precedence ---


@pytest.mark.asyncio
async def test_chain_jwt_token_trust_token_bearer_still_works(
    tmp_path, monkeypatch
) -> None:
    token = await _seed_user(tmp_path, "alice")
    config = Config(
        provider="anthropic",
        model="test-model",
        server=_server_config(
            ("jwt", "token", "trust"),
            issuer="https://test.example.com",
            jwks_url="https://test.example.com/.well-known/jwks.json",
        ),
    )
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        resp = await env["client"].get(
            "/v1/sessions", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_chain_jwt_token_trust_jwt_shape_does_not_fall_through_to_token(
    tmp_path, monkeypatch
) -> None:
    await _seed_user(tmp_path, "alice")
    config = Config(
        provider="anthropic",
        model="test-model",
        server=_server_config(
            ("jwt", "token", "trust"),
            issuer="https://test.example.com",
            jwks_url="https://test.example.com/.well-known/jwks.json",
        ),
    )
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        # Two-dot shape is claimed by the jwt authenticator, which fails it.
        resp = await env["client"].get(
            "/v1/sessions", headers={"Authorization": "Bearer a.b.c"}
        )
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_chain_jwt_token_trust_no_credentials_returns_401(
    tmp_path, monkeypatch
) -> None:
    await _seed_user(tmp_path, "alice")
    config = Config(
        provider="anthropic",
        model="test-model",
        server=_server_config(
            ("jwt", "token", "trust"),
            issuer="https://test.example.com",
            jwks_url="https://test.example.com/.well-known/jwks.json",
        ),
    )
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        resp = await env["client"].get("/v1/sessions")
        assert resp.status_code == 401


# --- jwt mode ---


def _rsa_keypair() -> tuple[bytes, bytes]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


class _FakeJWKClient:
    def __init__(self, public_key: bytes) -> None:
        self.public_key = public_key

    def get_signing_key_from_jwt(self, token: str) -> Any:
        return type("SigningKey", (), {"key": self.public_key})()


def _patch_jwk_client(jwt_auth: JwtAuthenticator, public_key: bytes) -> None:
    client = _FakeJWKClient(public_key)

    async def _fake_get_jwk_client(jwt_module: Any) -> Any:
        return client

    jwt_auth._get_jwk_client = _fake_get_jwk_client


def _jwt_config() -> ServerConfig:
    return _server_config(
        ("jwt",),
        issuer="https://test.example.com",
        audience="test-audience",
        jwks_url="https://test.example.com/.well-known/jwks.json",
    )


def _sign_jwt(
    private_key: bytes, *, sub: str = "alice", aud: str = "test-audience", exp: int | None = None
) -> str:
    import jwt as jwt_module

    now = int(datetime.now(timezone.utc).timestamp())
    payload: dict[str, Any] = {"sub": sub, "iss": "https://test.example.com"}
    if aud is not None:
        payload["aud"] = aud
    if exp is None:
        exp = now + 300
    payload["exp"] = exp
    return jwt_module.encode(payload, private_key, algorithm="RS256")


async def _jwt_auth_instance(app: Any) -> JwtAuthenticator:
    chain: AuthChain = app.state.auth
    for auth in chain.authenticators:
        if isinstance(auth, JwtAuthenticator):
            return auth
    raise RuntimeError("No JwtAuthenticator in chain")


@pytest.mark.asyncio
async def test_jwt_valid_token_returns_200_and_sets_source(
    tmp_path, monkeypatch
) -> None:
    private_key, public_key = _rsa_keypair()
    config = Config(
        provider="anthropic", model="test-model", server=_jwt_config()
    )
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        jwt_auth = await _jwt_auth_instance(env["app"])
        _patch_jwk_client(jwt_auth, public_key)

        token = _sign_jwt(private_key)
        resp = await env["client"].get(
            "/v1/sessions", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 200

        store = SqliteStore(tmp_path)
        await store.initialize()
        user = await store.get_or_create_user("alice", source="jwt")
        await store.close()
        assert user["auth_source"] == "jwt"


@pytest.mark.asyncio
async def test_jwt_expired_token_returns_401(tmp_path, monkeypatch) -> None:
    private_key, public_key = _rsa_keypair()
    config = Config(
        provider="anthropic", model="test-model", server=_jwt_config()
    )
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        jwt_auth = await _jwt_auth_instance(env["app"])
        _patch_jwk_client(jwt_auth, public_key)

        now = int(datetime.now(timezone.utc).timestamp())
        token = _sign_jwt(private_key, exp=now - 120)
        resp = await env["client"].get(
            "/v1/sessions", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_jwt_wrong_audience_returns_401(tmp_path, monkeypatch) -> None:
    private_key, public_key = _rsa_keypair()
    config = Config(
        provider="anthropic", model="test-model", server=_jwt_config()
    )
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        jwt_auth = await _jwt_auth_instance(env["app"])
        _patch_jwk_client(jwt_auth, public_key)

        token = _sign_jwt(private_key, aud="other-audience")
        resp = await env["client"].get(
            "/v1/sessions", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_jwt_missing_exp_returns_401(tmp_path, monkeypatch) -> None:
    private_key, public_key = _rsa_keypair()
    config = Config(
        provider="anthropic", model="test-model", server=_jwt_config()
    )
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        jwt_auth = await _jwt_auth_instance(env["app"])
        _patch_jwk_client(jwt_auth, public_key)

        import jwt as jwt_module

        payload = {
            "sub": "alice",
            "iss": "https://test.example.com",
            "aud": "test-audience",
        }
        token = jwt_module.encode(payload, private_key, algorithm="RS256")
        resp = await env["client"].get(
            "/v1/sessions", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 401


# --- store unit tests ---


@pytest.fixture
async def sqlite_store(tmp_path):
    store = SqliteStore(tmp_path)
    await store.initialize()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_get_or_create_user_creates_with_source(sqlite_store) -> None:
    user = await sqlite_store.get_or_create_user("alice", source="trust")
    assert user["username"] == "alice"
    assert user["is_active"] is True
    assert user["auth_source"] == "trust"


@pytest.mark.asyncio
async def test_get_or_create_user_is_idempotent(sqlite_store) -> None:
    first = await sqlite_store.get_or_create_user("alice", source="trust")
    second = await sqlite_store.get_or_create_user("alice", source="trust")
    assert first["user_id"] == second["user_id"]
    assert first["auth_source"] == "trust"


@pytest.mark.asyncio
async def test_get_or_create_user_rejects_cross_source_username(
    sqlite_store,
) -> None:
    await sqlite_store.add_user("alice")

    with pytest.raises(Exception) as exc_info:
        await sqlite_store.get_or_create_user("alice", source="trust")
    assert "different authentication source" in str(exc_info.value)

    rows = await sqlite_store.list_users()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_get_or_create_user_revoked_stays_revoked(sqlite_store) -> None:
    await sqlite_store.add_user("alice")
    await sqlite_store.revoke_user("alice")
    user = await sqlite_store.get_or_create_user("alice", source="token")
    assert user["is_active"] is False


# --- sanitize_user_id unit tests ---


@pytest.mark.parametrize(
    "value",
    [
        "alice",
        "Alice",
        "alice@example.com",
        "a.b-c+d_e",
        "a" * 128,
        "user123",
    ],
)
def test_sanitize_user_id_accepts_valid(value: str) -> None:
    assert sanitize_user_id(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "a/b",
        "../etc",
        "a b",
        "-alice",
        ".alice",
        "a" * 200,
        "alice$",
        "alice#",
    ],
)
def test_sanitize_user_id_rejects_invalid(value: str) -> None:
    with pytest.raises(Exception) as exc_info:
        sanitize_user_id(value)
    assert exc_info.value.status_code == 400


# --- new security-hardening tests ---


@pytest.mark.asyncio
async def test_token_revoked_user_returns_401(tmp_path, monkeypatch) -> None:
    """A revoked user's still-valid harn_ token must be rejected."""
    token = await _seed_user(tmp_path, "alice")
    import aiosqlite

    async with aiosqlite.connect(str(server_db_path(tmp_path))) as conn:
        await revoke_user(conn, "alice")

    config = Config(provider="anthropic", model="test-model")
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        resp = await env["client"].get(
            "/v1/sessions", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 401
        assert "Invalid or revoked token" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_chain_jwt_token_trust_two_dot_token_detail_is_jwt(
    tmp_path, monkeypatch
) -> None:
    """Bearer a.b.c must be claimed by jwt mode, not fall through to token."""
    await _seed_user(tmp_path, "alice")
    config = Config(
        provider="anthropic",
        model="test-model",
        server=_server_config(
            ("jwt", "token", "trust"),
            issuer="https://test.example.com",
            audience="test-audience",
            jwks_url="https://test.example.com/.well-known/jwks.json",
        ),
    )
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        resp = await env["client"].get(
            "/v1/sessions", headers={"Authorization": "Bearer a.b.c"}
        )
        assert resp.status_code == 401
        assert "Invalid token" in resp.json()["detail"]
        assert "Invalid or revoked token" != resp.json()["detail"]


@pytest.mark.asyncio
async def test_cross_source_conflict_returns_409_and_preserves_user_id(
    tmp_path, monkeypatch
) -> None:
    """A token user must not be adoptable via trust mode."""
    token = await _seed_user(tmp_path, "alice")
    config = Config(
        provider="anthropic",
        model="test-model",
        server=_server_config(("trust",)),
    )
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        store = SqliteStore(tmp_path)
        await store.initialize()
        original = await store.get_user_by_username("alice")
        await store.close()

        resp = await env["client"].get(
            "/v1/sessions", headers={"X-User-Id": "alice"}
        )
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert "alice" in detail
        assert "different authentication source" in detail

        store = SqliteStore(tmp_path)
        await store.initialize()
        after = await store.get_user_by_username("alice")
        await store.close()
        assert after is not None
        assert after["user_id"] == original["user_id"]


@pytest.mark.asyncio
async def test_same_source_trust_reuses_user_id(tmp_path, monkeypatch) -> None:
    """Two trust requests for the same id create exactly one row."""
    config = Config(
        provider="anthropic",
        model="test-model",
        server=_server_config(("trust",)),
    )
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        resp1 = await env["client"].get(
            "/v1/sessions", headers={"X-User-Id": "alice"}
        )
        assert resp1.status_code == 200

        store = SqliteStore(tmp_path)
        await store.initialize()
        first = await store.get_user_by_username("alice")
        await store.close()

        resp2 = await env["client"].get(
            "/v1/sessions", headers={"X-User-Id": "alice"}
        )
        assert resp2.status_code == 200

        store = SqliteStore(tmp_path)
        await store.initialize()
        rows = await store.list_users()
        second = await store.get_user_by_username("alice")
        await store.close()

        assert len(rows) == 1
        assert first["user_id"] == second["user_id"]


@pytest.mark.asyncio
async def test_auto_provision_cap_returns_429(tmp_path, monkeypatch) -> None:
    """With max_auto_provisioned_users=1, the second new user gets 429."""
    config = Config(
        provider="anthropic",
        model="test-model",
        server=ServerConfig(
            auth_modes=("trust",),
            max_auto_provisioned_users=1,
        ),
    )
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        resp1 = await env["client"].get(
            "/v1/sessions", headers={"X-User-Id": "alice"}
        )
        assert resp1.status_code == 200

        resp2 = await env["client"].get(
            "/v1/sessions", headers={"X-User-Id": "bob"}
        )
        assert resp2.status_code == 429
        detail = resp2.json()["detail"]
        assert "cap reached" in detail.lower()
        assert "max_auto_provisioned_users" in detail


# --- config security tests ---


def _write_toml(tmp_path: Path, text: str, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    for key in (
        "HARNESS_SERVER_AUTH_MODES",
        "HARNESS_SERVER_JWT_ISSUER",
        "HARNESS_SERVER_JWT_AUDIENCE",
        "HARNESS_SERVER_JWT_ALGORITHMS",
        "HARNESS_SERVER_JWT_JWKS_URL",
        "HARNESS_SERVER_JWT_DISCOVERY_TIMEOUT_SECONDS",
        "HARNESS_SERVER_MAX_AUTO_PROVISIONED_USERS",
    ):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / "harness.toml").write_text(text, encoding="utf-8")


def test_config_jwt_algorithm_hs256_exits(tmp_path: Path, monkeypatch) -> None:
    _write_toml(
        tmp_path,
        "[server]\nauth_modes = ['jwt']\n\n[server.jwt]\n"
        "issuer = 'https://id.example.com'\naudience = 'my-api'\n"
        "algorithms = ['HS256']\n",
        monkeypatch,
    )
    with pytest.raises(SystemExit):
        resolve_config(tmp_path)


def test_config_jwt_without_audience_exits(tmp_path: Path, monkeypatch) -> None:
    _write_toml(
        tmp_path,
        "[server]\nauth_modes = ['jwt']\n\n[server.jwt]\n"
        "issuer = 'https://id.example.com'\n",
        monkeypatch,
    )
    with pytest.raises(SystemExit):
        resolve_config(tmp_path)


def test_config_jwt_jwks_url_without_issuer_exits(
    tmp_path: Path, monkeypatch
) -> None:
    _write_toml(
        tmp_path,
        "[server]\nauth_modes = ['jwt']\n\n[server.jwt]\n"
        "audience = 'my-api'\n"
        "jwks_url = 'https://id.example.com/.well-known/jwks'\n",
        monkeypatch,
    )
    with pytest.raises(SystemExit):
        resolve_config(tmp_path)


def test_config_discovery_timeout_zero_preserved(
    tmp_path: Path, monkeypatch
) -> None:
    _write_toml(
        tmp_path,
        "[server]\nauth_modes = ['jwt']\n\n[server.jwt]\n"
        "issuer = 'https://id.example.com'\naudience = 'my-api'\n"
        "discovery_timeout_seconds = 0\n",
        monkeypatch,
    )
    config = resolve_config(tmp_path)
    assert config.server.jwt.discovery_timeout_seconds == 0.0


def test_config_max_auto_provisioned_from_toml(
    tmp_path: Path, monkeypatch
) -> None:
    _write_toml(
        tmp_path,
        "[server]\nmax_auto_provisioned_users = 100\n",
        monkeypatch,
    )
    config = resolve_config(tmp_path)
    assert config.server.max_auto_provisioned_users == 100


def test_config_max_auto_provisioned_from_env(
    tmp_path: Path, monkeypatch
) -> None:
    _write_toml(tmp_path, "[server]\nmax_auto_provisioned_users = 100\n", monkeypatch)
    monkeypatch.setenv("HARNESS_SERVER_MAX_AUTO_PROVISIONED_USERS", "250")
    config = resolve_config(tmp_path)
    assert config.server.max_auto_provisioned_users == 250
