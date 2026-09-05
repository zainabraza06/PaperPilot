"""Persistence for retrieved papers.

Export needs the full record for a paper the user selected, and a search
response is not something the server keeps. Two ways to bridge that: have
the client post the papers back, or store them when they are retrieved.

Storing them wins. Posting them back means trusting the client's copy of a
record to generate a citation — a caller could send anything and get a
plausible-looking citation for a paper that does not exist. It also makes
every export request carry a payload proportional to the selection.

Papers are stored as their serialized model rather than shredded into
columns. The Paper model is the schema, it changes as stages are added,
and nothing here queries by field — search is the query engine, this is
a keyed store. The cost is that it cannot be filtered in SQL, which it
never needs to be.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from app.core.logging import get_logger
from app.models.paper import Paper

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id         TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_papers_updated ON papers (updated_at);
"""


class PaperStore:
    """A keyed store of papers seen by any search."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        if self._path.parent != Path():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(self._path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.executescript(_SCHEMA)
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.commit()

    def save_many(self, papers: Sequence[Paper]) -> None:
        """Upsert every paper from a result set.

        Always overwrites rather than inserting-if-absent: a later search
        may have merged more sources into the same work, so the newer
        record is the more complete one.
        """
        if not papers:
            return
        now = datetime.now(UTC).isoformat()
        rows = [(p.id, p.model_dump_json(), now) for p in papers]
        with self._lock:
            self._connection.executemany(
                "INSERT OR REPLACE INTO papers (id, payload, updated_at) VALUES (?, ?, ?)",
                rows,
            )
            self._connection.commit()

    def get_many(self, paper_ids: Sequence[str]) -> list[Paper]:
        """Return the stored papers for these ids, in the order requested.

        Unknown ids are skipped rather than raising: the caller compares
        what it asked for against what came back and reports the
        difference, which is more useful than a single failed request.
        """
        if not paper_ids:
            return []
        placeholders = ",".join("?" for _ in paper_ids)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT id, payload FROM papers WHERE id IN ({placeholders})",
                tuple(paper_ids),
            ).fetchall()

        found: dict[str, Paper] = {}
        for row in rows:
            try:
                found[row["id"]] = Paper.model_validate_json(row["payload"])
            except Exception:
                logger.warning("discarding unreadable stored paper %s", row["id"])
        # Requested order is the user's selection order, which is what the
        # exported file should preserve.
        return [found[pid] for pid in paper_ids if pid in found]

    def count(self) -> int:
        with self._lock:
            row = self._connection.execute("SELECT COUNT(*) AS n FROM papers").fetchone()
        return int(row["n"])

    def close(self) -> None:
        with self._lock:
            self._connection.close()
