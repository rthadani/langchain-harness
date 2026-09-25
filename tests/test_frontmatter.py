from pathlib import Path

import pytest

from langchain_harness.frontmatter import parse_file, parse_frontmatter


@pytest.mark.parametrize(
    ("text", "expected_meta", "expected_body"),
    [
        ("---\nname: foo\n---\nbody text", {"name": "foo"}, "body text"),
        ("---\nname: foo\n---\n\nbody text", {"name": "foo"}, "body text"),
        ("no frontmatter\nhere", {}, "no frontmatter\nhere"),
        ("---\nnot closed", {}, "---\nnot closed"),
    ],
)
def test_parse_frontmatter(text, expected_meta, expected_body):
    meta, body = parse_frontmatter(text)
    assert meta == expected_meta
    assert body == expected_body


def test_parse_file(tmp_path: Path) -> None:
    path = tmp_path / "agent.md"
    path.write_text("---\nname: example\ndescription: desc\n---\nbody", encoding="utf-8")
    meta, body = parse_file(path)
    assert meta == {"name": "example", "description": "desc"}
    assert body == "body"


def test_embedded_dashes_not_delimiter() -> None:
    text = "---\nname: example\n---\nsome --- markdown\n--- more"
    meta, body = parse_frontmatter(text)
    assert meta == {"name": "example"}
    assert body == "some --- markdown\n--- more"
