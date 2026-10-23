"""Tests for the [server] config section."""

from __future__ import annotations

from pathlib import Path

import pytest

from langchain_harness.config import Config, ServerConfig, resolve_config


def _clear_keys(monkeypatch) -> None:
    for key in ("HARNESS_MODEL", "HARNESS_PROVIDER", "HARNESS_BASE_URL", "HARNESS_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    for key in ("HARNESS_SERVER_HOST", "HARNESS_SERVER_PORT", "HARNESS_SERVER_ENABLE_SHELL"):
        monkeypatch.delenv(key, raising=False)
    for key in (
        "HARNESS_SERVER_AUTH_MODES",
        "HARNESS_SERVER_JWT_ISSUER",
        "HARNESS_SERVER_JWT_AUDIENCE",
        "HARNESS_SERVER_JWT_USER_CLAIM",
        "HARNESS_SERVER_JWT_ALGORITHMS",
        "HARNESS_SERVER_JWT_JWKS_URL",
        "HARNESS_SERVER_JWT_CLOCK_SKEW_SECONDS",
        "HARNESS_SERVER_JWT_DISCOVERY_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)


def test_server_defaults(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    config = resolve_config(tmp_path)
    assert config.server == ServerConfig()


def test_server_from_toml(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    (tmp_path / "harness.toml").write_text(
        "[server]\nhost = '0.0.0.0'\nport = 9000\nenable_shell = true\n",
        encoding="utf-8",
    )
    config = resolve_config(tmp_path)
    assert config.server.host == "0.0.0.0"
    assert config.server.port == 9000
    assert config.server.enable_shell is True


def test_server_env_overrides_toml(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    monkeypatch.setenv("HARNESS_SERVER_HOST", "10.0.0.1")
    monkeypatch.setenv("HARNESS_SERVER_PORT", "8080")
    monkeypatch.setenv("HARNESS_SERVER_ENABLE_SHELL", "true")
    (tmp_path / "harness.toml").write_text(
        "[server]\nhost = '0.0.0.0'\nport = 9000\nenable_shell = false\n",
        encoding="utf-8",
    )
    config = resolve_config(tmp_path)
    assert config.server.host == "10.0.0.1"
    assert config.server.port == 8080
    assert config.server.enable_shell is True


def test_server_enable_shell_env_false(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    monkeypatch.setenv("HARNESS_SERVER_ENABLE_SHELL", "false")
    (tmp_path / "harness.toml").write_text(
        "[server]\nenable_shell = true\n",
        encoding="utf-8",
    )
    config = resolve_config(tmp_path)
    assert config.server.enable_shell is False


def test_server_invalid_port_falls_back(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    monkeypatch.setenv("HARNESS_SERVER_PORT", "not-a-number")
    config = resolve_config(tmp_path)
    assert config.server.port == 8000


def test_server_auth_modes_default_is_token(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    config = resolve_config(tmp_path)
    assert config.server.auth_modes == ("token",)


def test_server_auth_modes_from_toml(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    (tmp_path / "harness.toml").write_text(
        "[server]\nauth_modes = ['jwt', 'trust']\n\n[server.jwt]\n"
        "issuer = 'https://id.example.com'\naudience = 'my-api'\n",
        encoding="utf-8",
    )
    config = resolve_config(tmp_path)
    assert config.server.auth_modes == ("jwt", "trust")


def test_server_auth_modes_env_overrides_toml(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    monkeypatch.setenv("HARNESS_SERVER_AUTH_MODES", "token,trust")
    (tmp_path / "harness.toml").write_text(
        "[server]\nauth_modes = ['jwt', 'trust']\n\n[server.jwt]\nissuer = 'https://id.example.com'\n",
        encoding="utf-8",
    )
    config = resolve_config(tmp_path)
    assert config.server.auth_modes == ("token", "trust")


def test_server_unknown_auth_mode_exits(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    (tmp_path / "harness.toml").write_text(
        "[server]\nauth_modes = ['oauth2']\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        resolve_config(tmp_path)


def test_server_duplicate_auth_modes_exits(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    (tmp_path / "harness.toml").write_text(
        "[server]\nauth_modes = ['token', 'token']\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        resolve_config(tmp_path)


def test_server_jwt_without_issuer_or_jwks_url_exits(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    (tmp_path / "harness.toml").write_text(
        "[server]\nauth_modes = ['jwt']\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        resolve_config(tmp_path)


def test_server_jwt_config_from_toml(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    (tmp_path / "harness.toml").write_text(
        "[server]\nauth_modes = ['jwt']\n\n[server.jwt]\n"
        "issuer = 'https://id.example.com'\n"
        "audience = 'my-api'\n"
        "user_claim = 'email'\n"
        "algorithms = ['RS256']\n"
        "jwks_url = 'https://id.example.com/jwks'\n"
        "clock_skew_seconds = 120\n"
        "discovery_timeout_seconds = 10.0\n",
        encoding="utf-8",
    )
    config = resolve_config(tmp_path)
    jwt = config.server.jwt
    assert jwt.issuer == "https://id.example.com"
    assert jwt.audience == "my-api"
    assert jwt.user_claim == "email"
    assert jwt.algorithms == ("RS256",)
    assert jwt.jwks_url == "https://id.example.com/jwks"
    assert jwt.clock_skew_seconds == 120
    assert jwt.discovery_timeout_seconds == 10.0


def test_server_jwt_config_defaults_preserved(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    (tmp_path / "harness.toml").write_text(
        "[server]\nauth_modes = ['jwt']\n\n[server.jwt]\n"
        "issuer = 'https://id.example.com'\naudience = 'my-api'\n",
        encoding="utf-8",
    )
    config = resolve_config(tmp_path)
    jwt = config.server.jwt
    assert jwt.audience == "my-api"
    assert jwt.user_claim == "sub"
    assert jwt.algorithms == ("RS256", "ES256", "PS256")
    assert jwt.jwks_url is None
    assert jwt.clock_skew_seconds == 60
    assert jwt.discovery_timeout_seconds == 5.0


def test_server_jwt_env_overrides_toml(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    monkeypatch.setenv("HARNESS_SERVER_JWT_ISSUER", "https://env.example.com")
    monkeypatch.setenv("HARNESS_SERVER_JWT_AUDIENCE", "env-api")
    monkeypatch.setenv("HARNESS_SERVER_JWT_USER_CLAIM", "preferred_username")
    monkeypatch.setenv("HARNESS_SERVER_JWT_ALGORITHMS", "ES256,PS256")
    monkeypatch.setenv("HARNESS_SERVER_JWT_JWKS_URL", "https://env.example.com/jwks")
    monkeypatch.setenv("HARNESS_SERVER_JWT_CLOCK_SKEW_SECONDS", "90")
    monkeypatch.setenv("HARNESS_SERVER_JWT_DISCOVERY_TIMEOUT_SECONDS", "15.5")
    (tmp_path / "harness.toml").write_text(
        "[server]\nauth_modes = ['jwt']\n\n[server.jwt]\n"
        "issuer = 'https://id.example.com'\n"
        "audience = 'my-api'\n"
        "user_claim = 'email'\n"
        "algorithms = ['RS256']\n"
        "jwks_url = 'https://id.example.com/jwks'\n"
        "clock_skew_seconds = 120\n"
        "discovery_timeout_seconds = 10.0\n",
        encoding="utf-8",
    )
    config = resolve_config(tmp_path)
    jwt = config.server.jwt
    assert jwt.issuer == "https://env.example.com"
    assert jwt.audience == "env-api"
    assert jwt.user_claim == "preferred_username"
    assert jwt.algorithms == ("ES256", "PS256")
    assert jwt.jwks_url == "https://env.example.com/jwks"
    assert jwt.clock_skew_seconds == 90
    assert jwt.discovery_timeout_seconds == 15.5
