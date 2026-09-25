"""Configuration resolution for the harness CLI."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    """Resolved harness model configuration."""

    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None


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

    return Config(
        provider=resolved_provider,
        model=resolved_model,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
    )
