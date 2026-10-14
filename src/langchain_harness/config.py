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
class PromptsConfig:
    """User-configurable prompt overrides for the harness.

    Each field maps to a prompt key in ``[prompts]`` in harness.toml.
    Values may be inline text or ``file:<relative-path>`` to load from a file.
    ``None`` means use the built-in default.
    """

    system: str | None = None
    summarize: str | None = None
    eval: str | None = None
    compact: str | None = None
    compact_nudge: str | None = None


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
class ServerConfig:
    """Configuration for the multi-user web API server."""

    host: str = "127.0.0.1"
    port: int = 8000
    enable_shell: bool = False


@dataclass(frozen=True)
class Config:
    """Resolved harness model configuration."""

    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)


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


def _resolve_prompt_value(value: Any, root: Path) -> str | None:
    """Resolve a prompt value from TOML: inline text or ``file:<path>`` reference."""
    if not isinstance(value, str) or not value.strip():
        return None
    stripped = value.strip()
    if stripped.startswith("file:"):
        rel = stripped[len("file:"):].strip()
        path = (root / rel).resolve()
        try:
            return path.read_text(encoding="utf-8").strip() or None
        except OSError as exc:
            logger.warning("Could not read prompt file %s: %s; using built-in default", path, exc)
            return None
    return stripped


def _resolve_prompts(toml_data: dict[str, Any], root: Path) -> PromptsConfig:
    """Build a ``PromptsConfig`` from the ``[prompts]`` TOML section."""
    section = toml_data.get("prompts", {})
    if not isinstance(section, dict):
        section = {}
    return PromptsConfig(
        system=_resolve_prompt_value(section.get("system"), root),
        summarize=_resolve_prompt_value(section.get("summarize"), root),
        eval=_resolve_prompt_value(section.get("eval"), root),
        compact=_resolve_prompt_value(section.get("compact"), root),
        compact_nudge=_resolve_prompt_value(section.get("compact_nudge"), root),
    )


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


def _resolve_server(toml_data: dict[str, Any]) -> ServerConfig:
    """Build a ``ServerConfig`` from [server] TOML and env vars."""
    toml_section = toml_data.get("server", {})
    if not isinstance(toml_section, dict):
        toml_section = {}

    host = os.environ.get("HARNESS_SERVER_HOST") or toml_section.get("host") or "127.0.0.1"

    port_env = os.environ.get("HARNESS_SERVER_PORT")
    port_val = int(port_env) if port_env and port_env.isdigit() else None
    port = port_val if port_val and port_val > 0 else int(toml_section.get("port", 8000))

    shell_env = os.environ.get("HARNESS_SERVER_ENABLE_SHELL", "").lower()
    if shell_env in {"1", "true", "yes"}:
        enable_shell = True
    elif shell_env in {"0", "false", "no"}:
        enable_shell = False
    else:
        enable_shell = bool(toml_section.get("enable_shell", False))

    return ServerConfig(host=host, port=port, enable_shell=enable_shell)


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
    server = _resolve_server(toml)
    prompts = _resolve_prompts(toml, cwd)

    return Config(
        provider=resolved_provider,
        model=resolved_model,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        compaction=compaction,
        server=server,
        prompts=prompts,
    )
