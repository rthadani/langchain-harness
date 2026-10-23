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


# Auth modes the server knows how to build. Order in ``auth_modes`` is the
# order they are tried; the first one holding a credential for the request wins.
KNOWN_AUTH_MODES = ("token", "trust", "jwt")

# Asymmetric JWT algorithms only. HMAC (HS*) and 'none' are forbidden because
# the verifier uses the issuer's public JWKS: an attacker who can ask for a
# token signed with a shared secret would be able to forge tokens.
ALLOWED_JWT_ALGORITHMS = frozenset(
    {"RS256", "RS384", "RS512", "ES256", "ES384", "ES512", "PS256", "PS384", "PS512", "EdDSA"}
)


@dataclass(frozen=True)
class JwtConfig:
    """OIDC/JWT verification settings used when ``jwt`` is an auth mode.

    Signing keys come from the issuer's JWKS, discovered through
    ``<issuer>/.well-known/openid-configuration`` unless ``jwks_url`` overrides
    it. Only asymmetric algorithms are supported, so there is no shared secret
    to leak or rotate by hand.
    """

    issuer: str | None = None
    audience: str | None = None
    user_claim: str = "sub"
    algorithms: tuple[str, ...] = ("RS256", "ES256", "PS256")
    jwks_url: str | None = None
    clock_skew_seconds: int = 60
    discovery_timeout_seconds: float = 5.0


@dataclass(frozen=True)
class ServerConfig:
    """Configuration for the multi-user web API server."""

    host: str = "127.0.0.1"
    port: int = 8000
    enable_shell: bool = False
    auth_modes: tuple[str, ...] = ("token",)
    jwt: JwtConfig = field(default_factory=JwtConfig)
    max_auto_provisioned_users: int = 500


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


def _split_list(value: Any) -> tuple[str, ...] | None:
    """Parse a list (TOML array or comma-separated string)."""
    if value is None:
        return None
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, list | tuple):
        items = [str(item) for item in value]
    else:
        return None
    items = tuple(item.strip() for item in items if item.strip())
    return items or None


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _parse_max_auto_provisioned(value: Any) -> int | None:
    """Parse max_auto_provisioned_users; any integer is valid (<=0 is unlimited)."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_jwt(toml_data: dict[str, Any]) -> JwtConfig:
    """Build a ``JwtConfig`` from the ``[server.jwt]`` TOML section."""
    server_section = toml_data.get("server", {})
    if not isinstance(server_section, dict):
        server_section = {}
    section = server_section.get("jwt", {})
    if not isinstance(section, dict):
        section = {}

    defaults = JwtConfig()
    clock_skew = _parse_int(
        os.environ.get("HARNESS_SERVER_JWT_CLOCK_SKEW_SECONDS")
        or section.get("clock_skew_seconds")
    )
    algorithms_raw = _split_list(
        os.environ.get("HARNESS_SERVER_JWT_ALGORITHMS") or section.get("algorithms")
    )
    algorithms = (
        tuple(algo.upper() for algo in algorithms_raw)
        if algorithms_raw
        else defaults.algorithms
    )
    bad_algorithms = [algo for algo in algorithms if algo not in ALLOWED_JWT_ALGORITHMS]
    if bad_algorithms:
        print(
            f"Error: [server.jwt] algorithms contains {bad_algorithms[0]!r}, "
            "which is not an allowed algorithm. Only asymmetric algorithms are "
            f"permitted: {', '.join(sorted(ALLOWED_JWT_ALGORITHMS))}. "
            "HMAC (HS*) and 'none' are never allowed.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    discovery_timeout = _parse_float(
        os.environ.get("HARNESS_SERVER_JWT_DISCOVERY_TIMEOUT_SECONDS")
        or section.get("discovery_timeout_seconds")
    )

    return JwtConfig(
        issuer=os.environ.get("HARNESS_SERVER_JWT_ISSUER")
        or section.get("issuer")
        or defaults.issuer,
        audience=os.environ.get("HARNESS_SERVER_JWT_AUDIENCE")
        or section.get("audience")
        or defaults.audience,
        user_claim=os.environ.get("HARNESS_SERVER_JWT_USER_CLAIM")
        or section.get("user_claim")
        or defaults.user_claim,
        algorithms=algorithms,
        jwks_url=os.environ.get("HARNESS_SERVER_JWT_JWKS_URL")
        or section.get("jwks_url")
        or defaults.jwks_url,
        clock_skew_seconds=defaults.clock_skew_seconds if clock_skew is None else clock_skew,
        discovery_timeout_seconds=defaults.discovery_timeout_seconds
        if discovery_timeout is None
        else discovery_timeout,
    )


def _resolve_auth_modes(toml_data: dict[str, Any]) -> tuple[tuple[str, ...], JwtConfig]:
    """Resolve ``[server] auth_modes`` and validate the combination.

    Returns the validated modes and the ``JwtConfig`` resolved from the same
    TOML section so the caller does not resolve it twice.

    Raises ``SystemExit`` for an unknown mode, or for ``jwt`` without the
    required ``issuer`` and ``audience``: deployment mistakes that would
    otherwise surface as 401s.
    """
    server_section = toml_data.get("server", {})
    if not isinstance(server_section, dict):
        server_section = {}

    modes_raw = _split_list(
        os.environ.get("HARNESS_SERVER_AUTH_MODES")
        or server_section.get("auth_modes")
    )
    modes = tuple(mode.lower() for mode in modes_raw) if modes_raw else ("token",)

    unknown = [mode for mode in modes if mode not in KNOWN_AUTH_MODES]
    if unknown:
        print(
            f"Error: unknown [server] auth_modes entry {unknown[0]!r}. "
            f"Known modes: {', '.join(KNOWN_AUTH_MODES)}.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if len(set(modes)) != len(modes):
        print("Error: [server] auth_modes contains duplicates.", file=sys.stderr)
        raise SystemExit(1)

    jwt = _resolve_jwt(toml_data)
    if "jwt" in modes:
        missing = []
        if not jwt.issuer:
            missing.append("issuer")
        if not jwt.audience:
            missing.append("audience")
        if missing:
            print(
                f"Error: [server] auth_modes includes 'jwt' but [server.jwt] "
                f"is missing required key(s): {', '.join(missing)}.",
                file=sys.stderr,
            )
            raise SystemExit(1)

    return modes, jwt


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

    auth_modes, jwt = _resolve_auth_modes(toml_data)

    max_auto_env = _parse_max_auto_provisioned(
        os.environ.get("HARNESS_SERVER_MAX_AUTO_PROVISIONED_USERS")
    )
    max_auto_toml = _parse_max_auto_provisioned(
        toml_section.get("max_auto_provisioned_users")
    )
    max_auto_provisioned_users = (
        max_auto_env
        if max_auto_env is not None
        else (max_auto_toml if max_auto_toml is not None else 500)
    )

    return ServerConfig(
        host=host,
        port=port,
        enable_shell=enable_shell,
        auth_modes=auth_modes,
        jwt=jwt,
        max_auto_provisioned_users=max_auto_provisioned_users,
    )


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
