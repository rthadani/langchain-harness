"""Built-in prompt defaults, loaded from .txt files in this package directory."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DIR = Path(__file__).parent


def load(name: str) -> str:
    """Return the content of prompts/<name>.txt, stripped of surrounding whitespace."""
    path = _DIR / f"{name}.txt"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning("Could not load built-in prompt %r: %s", name, exc)
        return ""
