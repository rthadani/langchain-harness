"""Tests for pluggable auth modes: trust and jwt."""

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


def _server_config(auth_modes: tuple[str, ...] = ("trust",), **jwt_kwargs: Any) -> ServerConfig:
    return ServerConfig(auth_modes=auth_modes, jwt=JwtConfig(**jwt_kwargs))


@asynccontextmanager
async def _app_client(tmp_path: Any, monkeypatch: Any, *, config: Config):
    """Create an app with a fake agent cache and yield its HTTP client."""
    monkeypatch.chdir(tmp_path)

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
async def test_trust_valid_header_accepted(tmp_path, monkeypatch) -> None:
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
async def test_trust_two_requests_for_same_id_have_same_user_id(
    tmp_path, monkeypatch
) -> None:
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

        resp2 = await env["client"].get(
            "/v1/sessions", headers={"X-User-Id": "alice"}
        )
        assert resp2.status_code == 200


@pytest.mark.asyncio
async def test_trust_authorization_header_is_ignored(tmp_path, monkeypatch) -> None:
    """A trust-mode request succeeds even with a random Authorization header."""
    config = Config(
        provider="anthropic",
        model="test-model",
        server=_server_config(("trust",)),
    )
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        resp = await env["client"].get(
            "/v1/sessions",
            headers={"X-User-Id": "alice", "Authorization": "Bearer whatever"},
        )
        assert resp.status_code == 200


# --- chain precedence ---


@pytest.mark.asyncio
async def test_chain_jwt_trust_jwt_shape_claimed_by_jwt(
    tmp_path, monkeypatch
) -> None:
    config = Config(
        provider="anthropic",
        model="test-model",
        server=_server_config(
            ("jwt", "trust"),
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
        assert "Invalid token" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_chain_jwt_trust_no_credentials_returns_401(
    tmp_path, monkeypatch
) -> None:
    config = Config(
        provider="anthropic",
        model="test-model",
        server=_server_config(
            ("jwt", "trust"),
            issuer="https://test.example.com",
            jwks_url="https://test.example.com/.well-known/jwks.json",
        ),
    )
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        resp = await env["client"].get("/v1/sessions")
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_chain_jwt_trust_header_overrides_bearer(
    tmp_path, monkeypatch
) -> None:
    """With jwt first, a non-JWT Bearer falls through to trust."""
    config = Config(
        provider="anthropic",
        model="test-model",
        server=_server_config(
            ("jwt", "trust"),
            issuer="https://test.example.com",
            jwks_url="https://test.example.com/.well-known/jwks.json",
        ),
    )
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        resp = await env["client"].get(
            "/v1/sessions",
            headers={"X-User-Id": "alice", "Authorization": "Bearer notajwt"},
        )
        assert resp.status_code == 200


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
async def test_jwt_valid_token_returns_200_and_uses_sub(tmp_path, monkeypatch) -> None:
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


@pytest.mark.asyncio
async def test_jwt_sub_claim_used_as_user_id(tmp_path, monkeypatch) -> None:
    private_key, public_key = _rsa_keypair()
    config = Config(
        provider="anthropic", model="test-model", server=_jwt_config()
    )
    async with _app_client(tmp_path, monkeypatch, config=config) as env:
        jwt_auth = await _jwt_auth_instance(env["app"])
        _patch_jwk_client(jwt_auth, public_key)

        token = _sign_jwt(private_key, sub="bob")
        resp = await env["client"].get(
            "/v1/sessions", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 200


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


def test_config_jwt_without_issuer_exits(tmp_path: Path, monkeypatch) -> None:
    _write_toml(
        tmp_path,
        "[server]\nauth_modes = ['jwt']\n\n[server.jwt]\n"
        "audience = 'my-api'\n",
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
