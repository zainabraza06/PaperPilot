"""Settings parsing, focused on the values a deploy actually sets.

Most of `Settings` is plain defaults and needs no test. `cors_origins` is
different: it is the one setting that *must* be changed to deploy the
frontend and backend on separate hosts, and the natural way to write it
used to crash the process at import.
"""

from __future__ import annotations

import pytest

from app.config import Settings


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # The obvious thing to type, and the one that used to crash.
        ("https://paperpilot.vercel.app", ["https://paperpilot.vercel.app"]),
        # Several origins: a preview deployment alongside production.
        (
            "https://a.vercel.app,https://b.vercel.app",
            ["https://a.vercel.app", "https://b.vercel.app"],
        ),
        # Spaces around the commas are what a human writes.
        (
            "https://a.vercel.app, https://b.vercel.app",
            ["https://a.vercel.app", "https://b.vercel.app"],
        ),
        # The JSON form still works: NoDecode turned the decoder off for
        # everyone, so the validator has to handle this itself. Anyone who
        # already set it correctly must not be broken by the fix.
        ('["https://a.vercel.app"]', ["https://a.vercel.app"]),
        ('["https://a.app", "https://b.app"]', ["https://a.app", "https://b.app"]),
        # Trailing separators and blank entries are typos, not origins.
        ("https://a.app,", ["https://a.app"]),
        ("  https://a.app  ", ["https://a.app"]),
    ],
)
def test_cors_origins_accepts_the_forms_a_deploy_produces(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: list[str]
) -> None:
    monkeypatch.setenv("PAPERPILOT_CORS_ORIGINS", raw)
    assert Settings().cors_origins == expected


def test_cors_origins_defaults_to_local_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset must stay usable: the app runs with no configuration at all."""
    monkeypatch.delenv("PAPERPILOT_CORS_ORIGINS", raising=False)
    assert Settings().cors_origins == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def test_a_single_origin_is_not_split_into_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards the failure mode a naive `list(value)` would introduce."""
    monkeypatch.setenv("PAPERPILOT_CORS_ORIGINS", "https://x.app")
    origins = Settings().cors_origins
    assert len(origins) == 1
    assert origins[0].startswith("https://")
