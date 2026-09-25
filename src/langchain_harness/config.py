"""Configuration resolution for the harness CLI."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompactionConfig:
    """User-configurable conversation compaction thresholds.

    Either ``trigger_tokens`` or ``trigger_messages`` may be set, but not both.
    When neither is set the harness leaves compaction to deepagents defaults.
    """

    trigger_tokens: int | None = None
    trigger_messages: int | None = None
    keep_messages: int = 20


@dataclass(frozen=True)
class Config:
    """Resolved harness model configuration."""

    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    compaction: CompactionConfig = field(default_factory=CompactionConfig)


def _load_toml(path: Path) -> dict[str, Any]:
    """Load the optional ``harness.toml`` file."""
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (tomllib.TOMLDecodeError, OSError) as exc:
        logger.warning("Malformed harness.toml %s: %s; ignoring", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _parse_int(value: Any) -> int | None:
    """Parse a config value as a positive integer, returning ``None`` on failure."""
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _resolve_compaction(toml_data: dict[str, Any]) -> CompactionConfig:
    """Build a ``CompactionConfig`` from [compaction] TOML and env vars.

    Raises ``SystemExit`` if both trigger types are configured.
    """
    toml_section = toml_data.get("compaction", {})
    if not isinstance(toml_section, dict):
        toml_section = {}

    trigger_tokens = (
        _parse_int(os.environ.get("HARNESS_COMPACTION_TRIGGER_TOKENS"))
        or _parse_int(toml_section.get("trigger_tokens"))
    )
    trigger_messages = (
        _parse_int(os.environ.get("HARNESS_COMPACTION_TRIGGER_MESSAGES"))
        or _parse_int(toml_section.get("trigger_messages"))
    )
    keep_messages = (
        _parse_int(os.environ.get("HARNESS_COMPACTION_KEEP_MESSAGES"))
        or _parse_int(toml_section.get("keep_messages"))
        or 20
    )

    if trigger_tokens is not None and trigger_messages is not None:
        print(
            "Error: [compaction] trigger_tokens and trigger_messages are mutually exclusive.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    return CompactionConfig(
        trigger_tokens=trigger_tokens,
        trigger_messages=trigger_messages,
        keep_messages=keep_messages,
    )


def _api_key_fallback(provider: str) -> str | None:
    """Fall back to known provider-specific API key env vars."""
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY")
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    return None


def resolve_config(
    cwd: Path,
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Config:
    """Resolve configuration from env > CLI flags > ``harness.toml``.

    Raises ``SystemExit`` with a helpful message if the model is not set.
    """
    toml = _load_toml(cwd / "harness.toml")
    toml_model = toml.get("model", {}) if isinstance(toml.get("model"), dict) else {}

    resolved_provider = (
        os.environ.get("HARNESS_PROVIDER")
        or provider
        or toml_model.get("provider")
        or "anthropic"
    )

    resolved_model = (
        os.environ.get("HARNESS_MODEL")
        or model
        or toml_model.get("model")
    )

    resolved_base_url = (
        os.environ.get("HARNESS_BASE_URL")
        or base_url
        or toml_model.get("base_url")
    )

    resolved_api_key = (
        os.environ.get("HARNESS_API_KEY")
        or api_key
        or toml_model.get("api_key")
        or _api_key_fallback(resolved_provider)
    )

    if not resolved_model:
        print(
            "Error: no model configured. Set HARNESS_MODEL, pass --model, "
            "or add [model] model = '...' to harness.toml.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    compaction = _resolve_compaction(toml)

    return Config(
        provider=resolved_provider,
        model=resolved_model,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        compaction=compaction,
    )
