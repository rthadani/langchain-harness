"""Tests share SQLite by default; Postgres is opt-in per test."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _sqlite_by_default(monkeypatch):
    """Keep a developer's HARNESS_DATABASE_URL from reaching the test suite."""
    monkeypatch.delenv("HARNESS_DATABASE_URL", raising=False)
