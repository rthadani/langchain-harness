"""Minimal YAML frontmatter parser for skill and agent markdown files."""

from __future__ import annotations

import re
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
    if not stripped.startswith("---\n") and not stripped.startswith("---\r\n"):
        return {}, text

    # Split after the first --- line, then look for a matching closing line.
    _, rest = stripped.split("---", 1)
    close_match = re.search(r"^---\s*$", rest, flags=re.MULTILINE)
    if not close_match:
        return {}, text

    yaml_text = rest[: close_match.start()]
    body = rest[close_match.end() :]
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
