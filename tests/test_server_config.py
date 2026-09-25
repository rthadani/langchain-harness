"""Tests for the [server] config section."""

from __future__ import annotations

from pathlib import Path

from langchain_harness.config import Config, ServerConfig, resolve_config


def _clear_keys(monkeypatch) -> None:
    for key in ("HARNESS_MODEL", "HARNESS_PROVIDER", "HARNESS_BASE_URL", "HARNESS_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    for key in ("HARNESS_SERVER_HOST", "HARNESS_SERVER_PORT", "HARNESS_SERVER_ENABLE_SHELL"):
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
