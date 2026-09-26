"""Short-lived cache for whole search responses.

The summary cache saves the expensive *per-paper* work. This saves the
rest: the three-source fan-out, deduplication, ranking, clustering and
entity extraction all run again on a repeated query, and the fan-out alone
is 3-7 seconds because it is bounded by the slowest upstream API.

Three reasons this exists, in the order they matter:

* **The APIs ask for it.** NCBI, Crossref and arXiv all request that
  clients cache and avoid re-requesting identical data. arXiv in
  particular rate-limits by IP and its block outlasts the documented
  one-request-per-three-seconds window, so a deployment on a shared cloud
  address that re-fetches on every keystroke-adjacent search is asking to
  be cut off.
* **The literature does not change in an hour.** A search for a topic
  returns the same papers minute to minute. Paying seven seconds to
  rediscover that is not freshness, it is waste.
* **It makes the app feel like it works.** First impressions of a demo are
  decided by the first query, and 3-7 seconds is the difference between
  "slow" and "instant".

**What is deliberately not cached: a degraded response.** If arXiv timed
out, storing that result for an hour turns one bad minute into a bad hour,
and every user in that window sees a two-source answer with no way to
retry into a good one. Only responses where every source succeeded are
stored, which means a transient upstream failure costs exactly as long as
it lasts.

The key includes a *configuration fingerprint* for the same reason the
summary cache keys on ``prompt_version``: a response produced under a
different ranking strategy or a different summary model is a different
answer, and serving it after a config change would silently undo the
change.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from app.core.logging import get_logger
from app.models.paper import SourceName
from app.models.search import SearchResponse

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS search_responses (
    key        TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_search_created ON search_responses (created_at);
"""


def cache_key(
    query: str,
    limit_per_source: int,
    sources: list[SourceName] | None,
    config_fingerprint: str,
) -> str:
    """Digest of everything that changes what a search returns.

    The query is matched on its whitespace-collapsed, case-folded form, so
    a re-typed query with different spacing still hits. It is deliberately
    *not* normalized further than that — stemming or stopword-stripping
    here would merge queries the ranker treats as genuinely different.
    """
    normalized = " ".join(query.split()).casefold()
    names = ",".join(sorted(s.value for s in sources)) if sources else "all"
    digest = hashlib.sha256()
    for part in (normalized, str(limit_per_source), names, config_fingerprint):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:32]


class SearchCache:
    """SQLite-backed store of recent search responses.

    Synchronous, like the other stores here; callers run it in a worker
    thread. One connection under a lock is ample for this write volume.
    """

    def __init__(self, path: Path | str, ttl_seconds: int) -> None:
        self._path = Path(path)
        if self._path.parent != Path():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(self._path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.executescript(_SCHEMA)
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.commit()
        logger.info("search cache at %s (ttl %ss)", self._path, ttl_seconds)

    @property
    def ttl_seconds(self) -> int:
        return self._ttl

    def get(self, key: str) -> tuple[SearchResponse, int] | None:
        """Return the cached response and its age in seconds, if still fresh.

        An expired row is deleted on the way past rather than left for a
        sweeper: this is the only code that ever looks at it, so lazy
        expiry keeps the table small without a background task.
        """
        with self._lock:
            row = self._connection.execute(
                "SELECT payload, created_at FROM search_responses WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None

        try:
            created = datetime.fromisoformat(row["created_at"])
        except ValueError:
            self._delete(key)
            return None

        age = int((datetime.now(UTC) - created).total_seconds())
        if age >= self._ttl:
            self._delete(key)
            return None

        try:
            response = SearchResponse.model_validate_json(row["payload"])
        except Exception:
            # A schema change since the row was written. Dropping it is
            # correct and silent recovery; serving a half-parsed response
            # would be neither.
            logger.warning("discarding unreadable search cache row")
            self._delete(key)
            return None

        return response, age

    def put(self, key: str, response: SearchResponse) -> None:
        """Store a response. Callers are responsible for the degraded check."""
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO search_responses (key, payload, created_at) "
                "VALUES (?, ?, ?)",
                (key, response.model_dump_json(), datetime.now(UTC).isoformat()),
            )
            self._connection.commit()

    def _delete(self, key: str) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM search_responses WHERE key = ?", (key,))
            self._connection.commit()

    def count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS n FROM search_responses"
            ).fetchone()
        return int(row["n"])

    def clear(self) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM search_responses")
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()
