"""Minimal YAML frontmatter parser for skill and agent markdown files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter delimited by ``---`` from *text*.

    Returns a tuple of ``(metadata, body)``. If the file does not start with
    ``---``, metadata is returned as an empty dict and the original text is the
    body.
    """
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return {}, text

    # Split after the first --- line.
    _, rest = stripped.split("---", 1)
    parts = rest.split("---", 1)
    if len(parts) != 2:
        return {}, text

    yaml_text, body = parts
    try:
        metadata = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError:
        metadata = {}

    if not isinstance(metadata, dict):
        metadata = {}

    return metadata, body.lstrip("\n")


def parse_file(path: Path) -> tuple[dict[str, Any], str]:
    """Parse frontmatter from a file on disk."""
    return parse_frontmatter(path.read_text(encoding="utf-8"))
