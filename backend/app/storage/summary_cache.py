"""Persistent cache for generated summaries.

Summaries are the only expensive, non-deterministic and *chargeable* thing
the pipeline produces, and the same paper appears across repeated and
refined searches constantly. Regenerating it each time would cost money to
produce a worse answer, since a second sample is not more grounded than
the first — it is just different.

The cache key is the interesting part. It is not the paper id alone but
``(paper_id, model, prompt_version, source_fingerprint)``:

* **model** — text from a different model is a different answer.
* **prompt_version** — after a prompt is tightened, previously cached text
  was produced under the old instructions. Serving it would silently undo
  the fix.
* **source_fingerprint** — a digest of the title and abstract that were
  summarized. Deduplication merges records field-wise, so a paper that had
  no abstract on one search can have one on the next; its old summary
  described different input and must not be reused.

SQLite because it is the right size for this: one file, no service, safe
concurrent reads, and it survives a restart, which an in-process dict does
not.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from app.core.logging import get_logger
from app.models.summary import GroundingVerdict, Summary, SummaryOrigin

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS summaries (
    paper_id           TEXT    NOT NULL,
    model              TEXT    NOT NULL,
    prompt_version     INTEGER NOT NULL,
    source_fingerprint TEXT    NOT NULL,
    text               TEXT    NOT NULL,
    origin             TEXT    NOT NULL,
    grounding          TEXT    NOT NULL,
    attempts           INTEGER NOT NULL DEFAULT 1,
    created_at         TEXT    NOT NULL,
    PRIMARY KEY (paper_id, model, prompt_version, source_fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_summaries_created ON summaries (created_at);
"""


def fingerprint(title: str, abstract: str | None) -> str:
    """Digest of exactly the text that was summarized."""
    digest = hashlib.sha1()
    digest.update(title.encode("utf-8"))
    digest.update(b"\x00")
    digest.update((abstract or "").encode("utf-8"))
    return digest.hexdigest()[:16]


class SummaryCache:
    """A small SQLite-backed store of summaries and their verdicts.

    Methods are synchronous; callers run them in a worker thread. One
    connection is shared under a lock, which is ample for this write
    volume and avoids a connection pool for a single-file database.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        if self._path.parent != Path():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(self._path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.executescript(_SCHEMA)
            # Write-ahead logging lets reads proceed during a write, which
            # matters as soon as two searches overlap.
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.commit()
        logger.info("summary cache at %s", self._path)

    def get(
        self, paper_id: str, model: str, prompt_version: int, source_fingerprint: str
    ) -> Summary | None:
        """Return the cached summary for this exact input, if any."""
        with self._lock:
            row = self._connection.execute(
                "SELECT text, origin, grounding, attempts FROM summaries "
                "WHERE paper_id = ? AND model = ? AND prompt_version = ? "
                "AND source_fingerprint = ?",
                (paper_id, model, prompt_version, source_fingerprint),
            ).fetchone()
        if row is None:
            return None
        try:
            verdict = GroundingVerdict.model_validate_json(row["grounding"])
            origin = SummaryOrigin(row["origin"])
        except Exception:
            logger.warning("discarding unreadable cache row for %s", paper_id)
            return None
        return Summary(
            text=row["text"],
            origin=origin,
            grounding=verdict,
            # An extractive fallback is filed under the configured model but
            # was not produced by it; report the method that actually wrote
            # the text so the UI never mislabels it as AI-generated.
            model="extractive" if origin is SummaryOrigin.EXTRACTIVE else model,
            prompt_version=prompt_version,
            attempts=row["attempts"],
            cached=True,
        )

    def put(
        self,
        paper_id: str,
        model: str,
        prompt_version: int,
        source_fingerprint: str,
        summary: Summary,
    ) -> None:
        """Store a summary, replacing any entry for the same key.

        ``model`` is the *cache* key and is passed in rather than read off
        the summary, because the two can legitimately differ: when
        generation fails the grounding check, the stored text is extractive
        while the key must stay the configured model. Reading the key off
        ``summary.model`` would file that fallback under "extractive" and
        guarantee a miss on every future lookup.
        """
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO summaries "
                "(paper_id, model, prompt_version, source_fingerprint, text, origin, "
                " grounding, attempts, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    paper_id,
                    model,
                    prompt_version,
                    source_fingerprint,
                    summary.text,
                    summary.origin.value,
                    summary.grounding.model_dump_json(),
                    summary.attempts,
                    datetime.now(UTC).isoformat(),
                ),
            )
            self._connection.commit()

    def count(self) -> int:
        with self._lock:
            row = self._connection.execute("SELECT COUNT(*) AS n FROM summaries").fetchone()
        return int(row["n"])

    def clear(self) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM summaries")
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()
