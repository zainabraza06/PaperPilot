# PaperPilot

AI-powered scientific literature search across **PubMed**, **arXiv** and **Crossref** —
one query, one ranked list, AI summaries, one-click citation export.

> **Status: Stages 1–2 complete — multi-source retrieval and hybrid ranking,
> with measured Recall@k / NDCG@k on a hand-judged golden set.**
> Clustering (Stage 3), summarization (Stage 4), citation export (Stage 5),
> the React frontend (Stage 6) and packaging (Stage 7) are in progress.

---

## The problem

A researcher tracking a topic today searches PubMed, arXiv and Crossref separately.
Each has its own query syntax, its own response format, and its own gaps — PubMed has
no preprints, arXiv has no biomedical literature, Crossref has DOIs for everything but
abstracts for only some of it. The researcher then skims dozens of abstracts by hand
and formats citations by hand.

PaperPilot collapses that into one query, one ranked and deduplicated list.

---

## Stage 1 — the retrieval layer

One query fans out to three APIs concurrently, and the very different responses
(MEDLINE XML, an Atom feed, and JSON) are normalized into a single `Paper` model,
merged, and deduplicated.

### Try it

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements-dev.txt

# One real query against all three live APIs
python -m scripts.demo_search "prime editing in primary human cells"

# Compare ranking strategies on the golden set
python -m scripts.evaluate_ranking --per-query --sweep-alpha

# Or run the API
uvicorn app.main:app --reload
# → http://127.0.0.1:8000/docs
```

The first run downloads the embedding model (~90 MB). It is loaded once at startup,
in a worker thread, so no search pays for it.

No API keys or configuration are required. See [`backend/.env.example`](backend/.env.example)
for the optional keys that raise rate limits.

### What the demo shows

```
QUERY      10.1038/s41586-019-1711-4
INTENT     identifier (doi: 10.1038/s41586-019-1711-4)
==============================================================================

SOURCES
  [OK  ] PubMed      1 results   2530 ms
  [NONE] arXiv       0 results    873 ms  — arXiv returned no matches
  [OK  ] Crossref    1 results   1177 ms

MERGED     1 papers (1 duplicate record(s) collapsed) in 2530 ms
------------------------------------------------------------------------------
 1. Search-and-replace genome editing without double-strand breaks or …
    PubMed+Crossref  |  2019-10-21  |  Nature
    Andrew V Anzalone, Peyton B Randolph, Jessie R Davis +8 more
    DOI: 10.1038/s41586-019-1711-4
```

**On the timings:** PubMed is consistently the slowest source, and that is inherent to
its API rather than to this code. It is the only source that needs two sequential round
trips, and the second one — `efetch`, which returns full MEDLINE XML for every hit —
measures 3000–8400 ms on its own against 500–2300 ms for `esearch`. Our own rate limiter
contributes 400 ms of the gap between them when no API key is set. arXiv and Crossref are
single-call and land in 850–4300 ms; arXiv has measured *slower* than PubMed on some
queries. Retries contribute nothing: a three-query probe triggered zero. And because the
fan-out is concurrent, total search time is bounded by the slowest source, not their sum.

Three things worth noticing in that output:

- **`INTENT: identifier`** — the same search box accepts a topic, a keyword, a DOI /
  arXiv id / PMID, or a pasted abstract. A DOI triggers a direct lookup; a pasted
  abstract is distilled into keywords for the APIs while the full text is kept for
  semantic ranking in Stage 2.
- **`[NONE] arXiv`** — "arXiv has nothing on this" is a reported state, not an error.
  Every failure mode (timeout, rate limit, outage, unparseable response) surfaces the
  same way, so one flaky source can never break a search.
- **`PubMed+Crossref`** — the same work returned by two sources is merged into one
  record that carries the best fields from each.

---

## Stage 2 — ranking, and how it is measured

The retrieval layer returns dozens of papers in no meaningful order. Stage 2 scores
them against the query with two complementary signals and fuses the result.

**Semantic** — abstracts and the query are embedded with `all-MiniLM-L6-v2`
(sentence-transformers) and compared by cosine similarity. This is what matches
paraphrase: a query for "cell-free DNA screening" finds a paper that only ever says
"circulating tumour DNA".

**Lexical** — Okapi BM25 over the same text. This is what keeps rare, specific tokens
alive: a gene name, an assay, a model name. IDF rewards them explicitly, where a
bi-encoder smooths them toward their neighbourhood.

Both scorers see the *same* view of a paper ([`document.py`](backend/app/services/ranking/document.py)) —
title (repeated once, as a mild weighting), then abstract, or keywords and venue for
the ~half of Crossref records that have no abstract deposited.

The ranker is given the user's **raw** query, not the keyword string sent upstream.
That is the whole reason [`query_parser.py`](backend/app/services/query_parser.py)
preserves both: PubMed cannot accept a pasted 200-word abstract, but the embedding
model wants exactly that.

### Results

Eight queries, 174 candidates, **every candidate hand-judged** on a 0–3 scale.
Reproduce with `python -m scripts.evaluate_ranking`.

| strategy                     | R@5   | R@10  | R@20  | NDCG@5 | NDCG@10 | NDCG@20 | MRR   |
|------------------------------|-------|-------|-------|--------|---------|---------|-------|
| retrieval-order (no ranking) | 0.255 | 0.525 | 0.914 | 0.703  | 0.711   | 0.822   | 0.875 |
| lexical only (BM25)          | 0.296 | 0.580 | 0.976 | 0.798  | 0.843   | 0.921   | 0.906 |
| semantic only (embeddings)   | 0.347 | 0.620 | 0.976 | 0.877  | 0.887   | 0.943   | **1.000** |
| **hybrid linear (α=0.6)**    | 0.336 | **0.630** | 0.976 | 0.858 | **0.892** | 0.943 | 0.938 |
| hybrid RRF                   | 0.315 | 0.609 | **0.983** | 0.845 | 0.875 | 0.940 | 0.938 |

### How to read these numbers

**Recall@k has a ceiling below 1.0 here.** Most queries have more relevant papers than
*k*, and Recall@k cannot exceed `min(k, |relevant|) / |relevant|`. On this set the
ceilings are **R@5 ≤ 0.379, R@10 ≤ 0.675, R@20 ≤ 0.983**. So hybrid's 0.630 at k=10 is
**93% of the best any ranking could do**, not 63% of some ideal. Seven of the eight
queries hit 100% of their ceiling at k=10. The evaluator reports the ceiling on every
run so a correct number is not misread as a bad one.

**What is measured is re-ranking quality, not retrieval coverage.** The pool is the
complete retrieval output for each query and every candidate in it is judged, so there
is no top-k pooling bias and the "unjudged means irrelevant" assumption is doing no
hidden work. What these numbers do *not* say is how much of the wider literature the
fan-out found — that would need judgments over papers never returned.

**The honest read of the hybrid-vs-semantic comparison:** hybrid wins deeper in the
list (R@10, NDCG@10), semantic wins at the very top (R@5, NDCG@5, MRR), and the gaps
are ~0.01 on eight queries — inside the noise. The defensible claim is that *both
clearly beat doing nothing* (+0.105 R@10, +0.181 NDCG@10 over retrieval order) and that
**hybrid is the safer default because it degrades better**, not because it is
significantly more accurate on this set. An α sweep (0.2–0.8, in the evaluator) puts
the optimum at 0.6–0.7, which is where the default sits.

### What the evaluation caught

Building the golden set found three real defects that no unit test would have:

1. **PubMed silently returned zero results for ordinary topic queries.** PubMed ANDs
   every term, so `"CRISPR prime editing efficiency in human cells"` matched nothing
   while a shorter phrasing matched plenty. Both PubMed and arXiv now relax an empty
   ANDed query to OR — retrieval should favour recall, because precision is the
   ranker's job. That query went from 8 candidates to 24.
2. **Crossref peer-review records were poisoning the top of the ranking.** A
   `peer-review` record's title *quotes the reviewed paper's title*, so "Decision
   letter: Escape from neutralizing antibodies by SARS-CoV-2 spike…" scored top on
   both signals. On one query, all eight Crossref results were review reports. Record
   types are now filtered server-side. That query's NDCG@10 went **0.494 → 0.881** and
   its MRR **0.167 → 1.000**. A ranker cannot fix bad candidates; it can only reorder
   them.
3. **Preprints and their published versions appeared as separate results.** The
   conflicting-DOI veto (added in Stage 1 to keep errata separate) was also splitting
   the bioRxiv and journal versions of one paper — visibly, at ranks 1 and 3 of a demo.
   The veto now makes an exception for preprint DOI prefixes when titles match exactly.

### A measured idea that did not pay off

BM25 is a bag of words, so "prime editing" is scored as two independent terms and a
paper matching *editing* + *efficiency* + *CRISPR* + *human* can outrank one actually
about prime editing. Indexing adjacent token pairs fixes exactly that query
(NDCG@10 0.591 → 0.678) but **lowers the average** (0.892 → 0.870), because doubling
the term space dilutes unigram IDF everywhere else. It ships off by default, with the
trade-off recorded in [`lexical.py`](backend/app/services/ranking/lexical.py) rather
than quietly dropped.

### Known limitations

- **Eight queries is a small set.** Differences under ~0.02 should not be trusted.
- **BM25's IDF is local to the candidate set.** For a query like "prime editing" where
  every candidate mentions prime editing, the phrase carries almost no lexical weight.
  On a two-candidate set the lexical signal collapses to zero entirely (Okapi IDF is 0
  at df = N/2); the hybrid correctly falls back to semantic there.
- **Judgments are single-annotator**, assigned by reading each title and abstract
  against a documented rubric. There is no inter-annotator agreement figure.
- **Papers with no abstract rank on title and keywords alone**, which is genuinely
  weaker evidence — a known cost of Crossref's incomplete abstract coverage.
- The golden set is checked in ([`backend/eval/`](backend/eval/)) so the judgments can
  be audited: each entry carries its grade *and* the paper title.

### Degradation

Ranking is injected into the search service, never assumed. If the embedding model
cannot be downloaded, `build_embedder` falls back to a deterministic hashing embedder
and says so in `ranking.model`. If ranking raises, the search still returns its papers
in retrieval order with `ranking.applied = false` and a reason. The frontend can then
tell the user these results are unranked instead of presenting a worse list as if it
were ranked.

---

## Architecture

```
                        ┌──────────────────────────┐
   query ──────────────▶│      query_parser        │  topic / keyword /
                        │  intent classification   │  identifier / abstract
                        └────────────┬─────────────┘
                                     ▼
                        ┌──────────────────────────┐
                        │      SearchService       │  concurrent fan-out,
                        │   (per-source timeout)   │  per-source status
                        └────────────┬─────────────┘
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
      ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
      │  PubMedSource │      │  ArxivSource  │      │ CrossrefSource│
      │  MEDLINE XML  │      │   Atom feed   │      │     JSON      │
      └───────┬───────┘      └───────┬───────┘      └───────┬───────┘
              └──────────────────────┼──────────────────────┘
                                     ▼
                        ┌──────────────────────────┐
                        │        deduplicate       │  DOI match, then
                        │   (field-wise merge)     │  fuzzy title match
                        └────────────┬─────────────┘
                                     ▼
                        ┌──────────────────────────┐
                        │       HybridRanker       │  embeddings + BM25,
                        │   (fusion strategies)    │  fused and scored
                        └────────────┬─────────────┘
                                     ▼
                         ranked Paper[] + RelevanceScore
```

Every connector implements one interface:

```python
class PaperSource(abc.ABC):
    name: SourceName
    display_name: str

    async def search(self, query: SourceQuery) -> list[Paper]: ...
    async def fetch_by_doi(self, doi: str) -> Paper | None: ...
```

Adding a fourth provider (OpenAlex, Semantic Scholar) is a new file in
`app/sources/` plus one line in `app/sources/registry.py`. Nothing else in the
codebase names a concrete connector.

### Layout

```
backend/app/
├── api/            # thin routes + dependency wiring
├── core/           # text normalization, errors, logging, rate limiting, safe XML
├── models/         # Paper, Author, and the search request/response contract
├── services/       # query parsing, deduplication, search orchestration
│   └── ranking/    # document view, embeddings, BM25, fusion, IR metrics
└── sources/        # PaperSource interface + one module per provider

backend/eval/       # golden set: queries, frozen candidate pool, judgments
backend/scripts/    # demo_search, build_golden_set, evaluate_ranking
```

---

## Real-world quirks the connectors handle

These are the things that break a naive implementation, and each one is pinned by a test.

| Source | Quirk | Handling |
|---|---|---|
| PubMed | Two-call API (`esearch` → `efetch`) | Both calls behind one `search()`; relevance order restored after `efetch` |
| PubMed | Structured abstracts as several labelled nodes | Re-assembled as `Methods: …`, `Results: …` |
| PubMed | Inline `<i>`/`<sup>` markup inside titles and abstracts | Text gathered across child nodes, not read off `.text` |
| PubMed | Dates as Y/M/D, month names, seasons, or `"2021 Jan-Feb"` free text | Resolved in priority order, padded rather than dropped |
| PubMed | DOI in `ELocationID` *or* `ArticleIdList` | Both locations checked |
| PubMed | Free-text queries are ANDed, so specific topics match nothing | Empty result relaxes to an OR of the top terms |
| Crossref | `peer-review` records quote the reviewed paper's title and outrank it | Non-article types excluded server-side |
| Crossref | Preprint and published versions carry different DOIs | Merged when titles match exactly and one DOI is a preprint prefix |
| arXiv | ANDed multi-term queries match nothing on a preprint server | Same OR relaxation as PubMed |
| arXiv | Errors returned as **HTTP 200** with an error entry | Detected by entry id, raised as a parse error |
| arXiv | Versioned ids (`2101.00001v3`) | Version stripped so the id is stable |
| arXiv | 1 request / 3 seconds politeness policy | Per-source async rate limiter |
| Crossref | Abstracts deposited as JATS XML | Tags stripped, entities decoded |
| Crossref | Ragged `date-parts`: `[[2021]]`, `[[2021,5]]`, `[[null]]` | Padded, with fallback across `issued` / `published-print` / `published-online` |
| Crossref | Untitled stub records (datasets, components) | Dropped — nothing to rank or cite |
| All | Missing abstracts and DOIs are normal, not exceptional | Optional throughout; deduplication falls back to titles |
| All | Rate limits, 5xx, timeouts | Bounded retry with jitter + `Retry-After`; mapped to a per-source status |

XML from both XML sources is parsed with `defusedxml`, since it is untrusted input
from the public internet.

---

## Deduplication

The same paper legitimately appears in all three sources: an arXiv preprint gets a DOI
on publication, Crossref indexes the published version, PubMed indexes it again with
curated metadata.

1. **DOI match** — exact, after normalization (`https://doi.org/10.1/X` → `10.1/x`).
2. **Title match** — for the many records with no DOI, on a normalized title key
   (unaccented, lowercased, punctuation-folded), with a fuzzy fallback above a 0.94
   similarity ratio, blocked by title prefix to stay linear in practice.
3. **Conflicting DOIs veto a title match** — errata and corrections share their title
   with the paper they correct.

Merging is **field-wise, not winner-takes-all**: the most complete record keeps the
identity, and every field it lacks is filled from a sibling. That is how a Crossref
record with no abstract ends up carrying the arXiv abstract — which is exactly what
Stage 2's ranking needs.

---

## Testing

```bash
cd backend
python -m pytest          # 184 tests
python -m ruff check app tests
```

Tests are offline and deterministic. Connector parsing runs against recorded upstream
payloads in `tests/fixtures/`, chosen to include the awkward records (no abstract, no
DOI, collective authors, free-text dates, untitled stubs). Transport behaviour —
retries, error translation, the two-call PubMed flow — is tested against a mocked HTTP
layer with `respx`.

The IR metrics get particular attention: the README publishes numbers produced by
that module, so every expected value in its tests is hand-computed from the metric's
definition rather than captured from a previous run.

Coverage is concentrated where interviews probe:

| Area | File |
|---|---|
| PubMed normalization | `tests/test_pubmed_source.py` |
| arXiv normalization | `tests/test_arxiv_source.py` |
| Crossref normalization | `tests/test_crossref_source.py` |
| Deduplication (both false-merge and missed-merge directions) | `tests/test_dedupe.py` |
| Query classification | `tests/test_query_parser.py` |
| Fan-out and graceful degradation | `tests/test_search_service.py` |
| Retries, rate limits, timeouts | `tests/test_http_behaviour.py` |
| Embedders, BM25, and the fusion strategies | `tests/test_ranking.py` |
| IR metrics, hand-computed from their definitions | `tests/test_ranking_evaluation.py` |

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness plus the registered sources (the frontend builds its filters from this) |
| `GET` | `/api/search?q=…` | Search all sources, ranked, with per-source and ranking status |
| `POST` | `/api/search` | Same, for abstract snippets too long for a query string |
| `GET` | `/api/parse?q=…` | How a query would be interpreted — powers the live input-type hint in the UI |

Interactive docs at `/docs` when the server is running.

---

## Tech stack

**Backend** Python 3.12 · FastAPI · Pydantic v2 · httpx (async) · defusedxml
**Ranking** sentence-transformers (`all-MiniLM-L6-v2`) · rank-bm25 · NumPy
**Testing** pytest · pytest-asyncio · respx · ruff · mypy (strict)
**Coming** spaCy/SciSpacy + HDBSCAN (Stage 3) · grounded LLM summarization
(Stage 4) · React + TypeScript + Tailwind (Stage 6) · Docker Compose (Stage 7)
