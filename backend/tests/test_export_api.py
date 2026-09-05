"""Export endpoint and paper-store tests."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_paper_store
from app.config import Settings
from app.main import create_app
from app.models.paper import Author, Paper, SourceName
from app.storage.paper_store import PaperStore


def make_paper(paper_id: str = "doi:10.1/abc", title: str = "A study of things") -> Paper:
    return Paper(
        id=paper_id,
        doi="10.1/abc",
        source=SourceName.PUBMED,
        source_id="1",
        title=title,
        authors=[Author(name="Wei Chen", given="Wei", family="Chen")],
        abstract="An abstract.",
        published_date=date(2022, 7, 13),
        url="https://example.org/1",
        journal="Nature",
        publication_type="journal-article",
    )


@pytest.fixture
def store(tmp_path: Path) -> PaperStore:
    return PaperStore(tmp_path / "papers.db")


@pytest.fixture
def client(store: PaperStore) -> Iterator[TestClient]:
    app = create_app(Settings(environment="test", summaries_enabled=False))
    app.dependency_overrides[get_paper_store] = lambda: store
    with TestClient(app) as test_client:
        yield test_client


# --- the store ------------------------------------------------------------


def test_papers_round_trip_through_the_store(store: PaperStore) -> None:
    original = make_paper()
    store.save_many([original])
    restored = store.get_many([original.id])
    assert len(restored) == 1
    assert restored[0].title == original.title
    assert restored[0].doi == original.doi
    assert restored[0].published_date == original.published_date


def test_the_store_preserves_the_requested_order(store: PaperStore) -> None:
    # The request order is the user's selection order, which is what the
    # exported file should preserve.
    papers = [make_paper(f"p{n}", f"Study {n}") for n in range(4)]
    store.save_many(papers)
    restored = store.get_many(["p3", "p1", "p2"])
    assert [p.id for p in restored] == ["p3", "p1", "p2"]


def test_unknown_ids_are_skipped_rather_than_raising(store: PaperStore) -> None:
    store.save_many([make_paper("known")])
    assert [p.id for p in store.get_many(["known", "missing"])] == ["known"]


def test_saving_the_same_paper_again_overwrites_it(store: PaperStore) -> None:
    # A later search may have merged more sources into the same work, so
    # the newer record is the more complete one.
    store.save_many([make_paper("p1", "Original title")])
    store.save_many([make_paper("p1", "Enriched title")])
    assert store.count() == 1
    assert store.get_many(["p1"])[0].title == "Enriched title"


def test_saving_nothing_is_not_an_error(store: PaperStore) -> None:
    store.save_many([])
    assert store.count() == 0


def test_an_unreadable_row_is_skipped_not_fatal(store: PaperStore) -> None:
    store.save_many([make_paper("good")])
    store._connection.execute(
        "INSERT INTO papers (id, payload, updated_at) VALUES ('bad', '{not json', 'now')"
    )
    store._connection.commit()
    assert [p.id for p in store.get_many(["good", "bad"])] == ["good"]


# --- the endpoints --------------------------------------------------------


def test_bulk_export_returns_a_downloadable_file(
    client: TestClient, store: PaperStore
) -> None:
    store.save_many([make_paper(f"p{n}", f"Study {n}") for n in range(3)])
    response = client.post("/api/export", json={"paper_ids": ["p0", "p1", "p2"]})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-bibtex")
    assert "attachment" in response.headers["content-disposition"]
    assert response.text.count("@article") == 3


@pytest.mark.parametrize(
    ("export_format", "marker"),
    [
        ("bibtex", "@article{"),
        ("ris", "TY  - JOUR"),
        ("apa", "Chen, W. (2022)."),
        ("vancouver", "1. Chen W."),
    ],
)
def test_every_format_is_reachable_through_the_api(
    client: TestClient, store: PaperStore, export_format: str, marker: str
) -> None:
    store.save_many([make_paper("p1")])
    response = client.post(
        "/api/export", json={"paper_ids": ["p1"], "format": export_format}
    )
    assert response.status_code == 200
    assert marker in response.text


def test_single_paper_export_is_a_plain_get(client: TestClient, store: PaperStore) -> None:
    # A GET so the UI can make it an ordinary link.
    store.save_many([make_paper("p1")])
    response = client.get("/api/export/p1", params={"format": "ris"})
    assert response.status_code == 200
    assert response.text.startswith("TY  - ")


def test_a_partial_selection_still_exports_and_reports_the_gap(
    client: TestClient, store: PaperStore
) -> None:
    # Some ids may have aged out of the store; failing the whole export
    # would be worse than exporting what is there and saying so.
    store.save_many([make_paper("p1")])
    response = client.post("/api/export", json={"paper_ids": ["p1", "gone", "also-gone"]})
    assert response.status_code == 200
    assert response.headers["X-PaperPilot-Missing"] == "2"


def test_exporting_only_unknown_papers_is_a_404(client: TestClient) -> None:
    response = client.post("/api/export", json={"paper_ids": ["nope"]})
    assert response.status_code == 404


def test_an_unknown_single_paper_is_a_404(client: TestClient) -> None:
    assert client.get("/api/export/nope").status_code == 404


def test_an_empty_selection_is_rejected(client: TestClient) -> None:
    assert client.post("/api/export", json={"paper_ids": []}).status_code == 422


def test_an_oversized_selection_is_rejected(client: TestClient) -> None:
    # An unbounded id list is a cheap way to make the server read the whole
    # store into memory.
    response = client.post("/api/export", json={"paper_ids": [f"p{n}" for n in range(501)]})
    assert response.status_code == 422


def test_an_unknown_format_is_rejected(client: TestClient) -> None:
    response = client.post("/api/export", json={"paper_ids": ["p1"], "format": "endnote"})
    assert response.status_code == 422


def test_the_filename_is_sanitized(client: TestClient, store: PaperStore) -> None:
    # A filename reaches the browser inside a header; quotes and path
    # separators have no legitimate use there.
    store.save_many([make_paper("p1")])
    response = client.post(
        "/api/export",
        json={"paper_ids": ["p1"], "filename": '../../etc/passwd"; rm -rf /'},
    )
    disposition = response.headers["content-disposition"]
    assert ".." not in disposition
    assert "/" not in disposition
    assert disposition.count('"') == 2  # only the delimiters


def test_the_format_list_is_discoverable(client: TestClient) -> None:
    body = client.get("/api/export/formats/available").json()
    assert {entry["id"] for entry in body} == {"bibtex", "ris", "apa", "vancouver"}
    assert all(entry["label"] and entry["extension"] for entry in body)
