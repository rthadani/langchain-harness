from pathlib import Path

import pytest

from langchain_harness.config import CompactionConfig, Config, resolve_config


def _clear_keys(monkeypatch) -> None:
    for key in ("HARNESS_MODEL", "HARNESS_PROVIDER", "HARNESS_BASE_URL", "HARNESS_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def test_resolve_from_defaults(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    (tmp_path / "harness.toml").write_text(
        "[model]\nprovider = 'openai'\nmodel = 'gpt-4o'\n", encoding="utf-8"
    )
    config = resolve_config(tmp_path)
    assert config == Config(provider="openai", model="gpt-4o")


def test_env_overrides_toml(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    (tmp_path / "harness.toml").write_text(
        "[model]\nprovider = 'openai'\nmodel = 'gpt-4o'\n", encoding="utf-8"
    )
    monkeypatch.setenv("HARNESS_MODEL", "claude-sonnet")
    monkeypatch.setenv("HARNESS_PROVIDER", "anthropic")
    config = resolve_config(tmp_path, model="gpt-3")
    assert config == Config(provider="anthropic", model="claude-sonnet")


def test_flags_override_toml(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    (tmp_path / "harness.toml").write_text(
        "[model]\nprovider = 'anthropic'\nmodel = 'claude-3'\n", encoding="utf-8"
    )
    config = resolve_config(tmp_path, model="gpt-4")
    assert config == Config(provider="anthropic", model="gpt-4")


def test_api_key_fallback_anthropic(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setenv("HARNESS_MODEL", "claude-3")
    config = resolve_config(tmp_path)
    assert config.api_key == "secret"


def test_api_key_fallback_openai(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "open-secret")
    monkeypatch.setenv("HARNESS_PROVIDER", "openai")
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    config = resolve_config(tmp_path)
    assert config.api_key == "open-secret"


def test_missing_model_exits(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    (tmp_path / "harness.toml").write_text("[model]\nprovider = 'openai'\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        resolve_config(tmp_path)


def test_base_url_resolution(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    monkeypatch.setenv("HARNESS_BASE_URL", "http://example.com")
    config = resolve_config(tmp_path)
    assert config.base_url == "http://example.com"


def test_harness_api_key_beats_provider_fallback(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("HARNESS_API_KEY", "harness-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("HARNESS_MODEL", "claude-3")
    config = resolve_config(tmp_path)
    assert config.api_key == "harness-secret"


def test_malformed_toml_warns_and_returns_empty(tmp_path: Path, monkeypatch, caplog) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    (tmp_path / "harness.toml").write_text("[model\nprovider = 'openai'\n", encoding="utf-8")
    with caplog.at_level("WARNING"):
        config = resolve_config(tmp_path)
    assert config == Config(provider="anthropic", model="gpt-4")
    assert "Malformed harness.toml" in caplog.text


def test_compaction_defaults_to_unconfigured(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    config = resolve_config(tmp_path)
    assert config.compaction == CompactionConfig()


def test_compaction_parsed_from_toml(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    (tmp_path / "harness.toml").write_text(
        "[compaction]\ntrigger_tokens = 1000\nkeep_messages = 10\n",
        encoding="utf-8",
    )
    config = resolve_config(tmp_path)
    assert config.compaction.trigger_tokens == 1000
    assert config.compaction.trigger_messages is None
    assert config.compaction.keep_messages == 10


def test_compaction_env_overrides_toml(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    monkeypatch.setenv("HARNESS_COMPACTION_TRIGGER_TOKENS", "500")
    monkeypatch.setenv("HARNESS_COMPACTION_KEEP_MESSAGES", "5")
    (tmp_path / "harness.toml").write_text(
        "[compaction]\ntrigger_tokens = 1000\nkeep_messages = 10\n",
        encoding="utf-8",
    )
    config = resolve_config(tmp_path)
    assert config.compaction.trigger_tokens == 500
    assert config.compaction.keep_messages == 5


def test_compaction_mutual_exclusion_errors(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    (tmp_path / "harness.toml").write_text(
        "[compaction]\ntrigger_tokens = 1000\ntrigger_messages = 50\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        resolve_config(tmp_path)


def test_compaction_invalid_values_ignored(tmp_path: Path, monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-4")
    (tmp_path / "harness.toml").write_text(
        "[compaction]\ntrigger_tokens = 'lots'\nkeep_messages = -1\n",
        encoding="utf-8",
    )
    config = resolve_config(tmp_path)
    assert config.compaction.trigger_tokens is None
    assert config.compaction.keep_messages == 20
