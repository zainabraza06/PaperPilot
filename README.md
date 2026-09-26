# PaperPilot

[![CI](https://github.com/zainabraza06/PaperPilot/actions/workflows/ci.yml/badge.svg)](https://github.com/zainabraza06/PaperPilot/actions/workflows/ci.yml)

AI-powered scientific literature search across **PubMed**, **arXiv** and **Crossref** —
one query, one ranked list, AI summaries, one-click citation export.

> **Status: complete and measured.** Multi-source retrieval, hybrid ranking, NER and
> topic clustering, grounded summarization with a fact-checking layer, spec-correct
> citation export, a React frontend driven by a real browser, Docker packaging.

## What is actually interesting here

Every "AI-powered search" project claims relevance and summarization. This one
measures both, and reports the results that came out badly:

- **Ranking is evaluated on a hand-judged golden set** — 20 queries, 411 candidates,
  every one graded 0–3. The shipped improvement was picked by **paired bootstrap**:
  the biggest number in the sweep was NDCG@5 **+0.038** with a confidence interval of
  [−0.011, +0.105] on a 6–4 split, so it was discarded in favour of **+0.014
  [+0.004, +0.025]** on a 10–3 split. The small real effect, not the large noisy one.

- **The grounding check was blind to a whole class of fabrication**, and that was
  measured rather than assumed: claims recombined from the abstract's own sentences
  slipped past all six lexical rules **934 times out of 934**. A semantic support
  check took it to **61.3%** — still missing two in five, which is stated wherever
  the number appears.

- **Summary quality is compared against baselines, and the LLM loses one of them.**
  Generated summaries rewrite genuinely (70.6% novel bigrams against the extractive
  baseline's 1.9%) and read the whole abstract rather than its opening — and carry
  **~10% less of it** than simply taking three sentences. A prompt revision aimed at
  that gap moved coverage 0.658 → 0.658 and was reverted.

- **The palette is audited, not eyeballed.** A script scores all 24 colour pairings
  against WCAG 2.2 in both themes and **gates the build**. It caught three real
  failures on first run, including control borders at 1.5:1 where 1.4.11 asks for 3.

- **Four UI defects shipped past a green Playwright run** and were found by opening
  the screenshots — including a detail modal that rendered fully transparent in dark
  mode while `[role="dialog"]` was still, technically, visible.

Several measured ideas did not pay off and are kept as negative results: bigram BM25,
a coverage-focused prompt rewrite, and the honest reading that hybrid ranking beats
semantic-alone by about a point, which is inside the noise on 20 queries.

---


## The problem

A researcher tracking a topic today searches PubMed, arXiv and Crossref separately.
Each has its own query syntax, its own response format, and its own gaps — PubMed has
no preprints, arXiv has no biomedical literature, Crossref has DOIs for everything but
abstracts for only some of it. The researcher then skims dozens of abstracts by hand
and formats citations by hand.

PaperPilot collapses that into one query, one ranked and deduplicated list.

---

## Quickstart

### Docker (everything, one command)

```bash
docker compose up --build
# → http://localhost:5173
```

Runs with **no configuration at all**. The first build takes a few minutes:
the embedding model and the spaCy pipeline are baked into the image rather
than downloaded on first request, so the container starts offline and no
user pays for a download.

Every setting below is optional. Each one raises a limit or switches on a
better model, and the app reports which parts are degraded instead of
failing:

```bash
# .env, next to docker-compose.yml
PAPERPILOT_MISTRAL_API_KEY=      # AI summaries; without it they are extractive
PAPERPILOT_CROSSREF_MAILTO=      # puts Crossref calls in the faster polite pool
PAPERPILOT_PUBMED_API_KEY=       # raises NCBI from 3 to 10 requests/second
```

### Local development

Two terminals. The backend needs Python 3.12+, the frontend Node 20+.

```bash
# backend
cd backend
python -m venv .venv && .venv/Scripts/activate   # source .venv/bin/activate on macOS/Linux
pip install -r requirements-dev.txt
python -m spacy download en_core_web_sm
uvicorn app.main:app --reload                    # → http://127.0.0.1:8000/docs
```

```bash
# frontend
cd frontend
npm install
npm run dev                                      # → http://127.0.0.1:5173
```

If something already owns port 8000, point the dev proxy elsewhere rather
than editing a tracked file:

```bash
VITE_API_PROXY=http://127.0.0.1:8010 npm run dev
```

### Without the frontend

Every stage is exercisable from the command line, which is the fastest way
to see what the pipeline actually decides:

```bash
cd backend
python -m scripts.demo_search "CRISPR prime editing efficiency in human cells"
python -m scripts.demo_search "10.1038/s41586-019-1711-4"        # DOI lookup
python -m scripts.demo_search "prime editing" --export ris       # citations
python -m scripts.demo_search "diffusion models" --no-rank       # A/B the ranker

python -m scripts.evaluate_ranking --per-query --sweep-alpha --clusters
python -m scripts.evaluate_grounding
python -m scripts.evaluate_summaries --limit 20                  # needs a key
```

Regenerating the frontend's types after a backend model change:

```bash
cd backend && python -m scripts.dump_openapi
cd ../frontend && npx openapi-typescript openapi.json -o src/types/api.ts
```

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
              ┌──────────────────────┴──────────────────────┐
              ▼                                             ▼
      ┌───────────────────┐  ┌──────────────────┐  ┌───────────────────────┐
      │  EntityExtractor  │  │ PaperSummarizer  │  │    TopicClusterer     │
      │  (spaCy + rules)  │  │ generate → check │  │  (ward + c-TF-IDF)    │
      │                   │  │ → correct → fall │  │                       │
      └─────────┬─────────┘  └────────┬─────────┘  └───────────┬───────────┘
                └─────────────────────┼────────────────────────┘
                                      ▼
        ranked Paper[] + scores + entities + clusters + grounded summaries
                                      │
                                      ▼
                        ┌──────────────────────────┐
                        │        PaperStore        │  SQLite, so a selection
                        │   (id → enriched Paper)  │  can be cited later
                        └────────────┬─────────────┘
                                     ▼
                       BibTeX · RIS · APA · Vancouver
```

The embedder is shared between the ranker and the clusterer behind an LRU cache, so
a result set is embedded once per search rather than twice.

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
frontend/src/
├── components/     # search, results, clusters, detail modal, export, history
├── hooks/          # search lifecycle, theme, history, debounce
├── lib/            # the single API client
└── types/          # generated from OpenAPI + readable aliases

backend/app/
├── api/            # thin routes + dependency wiring
├── core/           # text normalization, errors, logging, rate limiting, safe XML
├── models/         # Paper, Author, and the search request/response contract
├── services/       # query parsing, deduplication, search orchestration
│   ├── enrichment/ # NER (spaCy + shape patterns) and topic clustering
│   ├── export/     # BibTeX, RIS, APA and Vancouver formatters
│   ├── ranking/    # document view, embeddings, cache, BM25, fusion, IR metrics
│   └── summarization/  # providers, prompts, grounding check, extractive fallback
├── storage/        # SQLite summary cache and paper store
└── sources/        # PaperSource interface + one module per provider

backend/eval/       # golden set: queries, frozen candidate pool, judgments
backend/scripts/    # demo_search, build_golden_set, evaluate_ranking
```

---

## Stage 1 — the retrieval layer

One query fans out to three APIs concurrently, and the very different responses
(MEDLINE XML, an Atom feed, and JSON) are normalized into a single `Paper` model,
merged, and deduplicated.

### What the CLI demo shows

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

**20 queries, 411 candidates, every candidate hand-judged** on a 0–3 scale.
Reproduce with `python -m scripts.evaluate_ranking`.

| strategy                     | R@5   | R@10  | R@20  | NDCG@5 | NDCG@10 | NDCG@20 | MRR   |
|------------------------------|-------|-------|-------|--------|---------|---------|-------|
| retrieval-order (no ranking) | 0.294 | 0.529 | 0.910 | 0.733  | 0.727   | 0.839   | 0.925 |
| lexical only (BM25)          | 0.333 | 0.604 | 0.971 | 0.861  | 0.864   | 0.933   | 0.942 |
| semantic only (embeddings)   | 0.349 | 0.621 | 0.972 | 0.906  | 0.899   | 0.950   | **0.975** |
| **hybrid linear (α=0.6)**    | 0.346 | **0.630** | 0.972 | **0.917** | **0.915** | **0.959** | **0.975** |
| hybrid RRF                   | 0.342 | 0.614 | **0.974** | 0.910 | 0.894 | 0.954 | **0.975** |
| ↳ minus the evidence prior   | 0.351 | 0.628 | 0.972 | 0.902  | 0.901   | 0.954   | 0.950 |

The golden set started at 8 queries and 174 candidates. That was too few to separate an
effect from noise, and it skewed biomedical and ML; the twelve added queries bring in
physics, public health, a three-way cross-domain intersection, and two input modes the
retrieval layer had never been scored on — a DataCite identifier and a second pasted
abstract. Judgments are keyed by paper id rather than pool position, so rebuilding the
pool cannot silently reassign a grade to a different paper.

### How to read these numbers

**Recall@k has a ceiling below 1.0 here.** Most queries have more relevant papers than
*k*, and Recall@k cannot exceed `min(k, |relevant|) / |relevant|`. On this set the
ceilings are **R@5 ≤ 0.376, R@10 ≤ 0.669, R@20 ≤ 0.991**. So hybrid's 0.630 at k=10 is
**94% of the best any ranking could do**, not 63% of some ideal. Fourteen of the twenty
queries hit 100% of their ceiling at k=10. The evaluator reports the ceiling on every
run so a correct number is not misread as a bad one.

**What is measured is re-ranking quality, not retrieval coverage.** The pool is the
complete retrieval output for each query and every candidate in it is judged, so there
is no top-k pooling bias and the "unjudged means irrelevant" assumption is doing no
hidden work. What these numbers do *not* say is how much of the wider literature the
fan-out found — that would need judgments over papers never returned.

**The honest read of the hybrid-vs-semantic comparison:** on 20 queries hybrid now
leads semantic on every metric rather than trading wins with it, but by 0.01–0.02 —
still small enough that the defensible claim is the same one as before. Both *clearly*
beat doing nothing (+0.101 R@10, +0.188 NDCG@10 over retrieval order), and **hybrid is
the safer default because it degrades better**, not because it is decisively more
accurate. An α sweep (0.2–0.8, in the evaluator) puts the optimum at 0.5–0.6, which is
where the default sits.

### What the evaluation caught

Building the golden set found four real defects that no unit test would have:

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
4. **Records with nothing but a title were out-ranking real papers.** Both scorers
   reward term density, and a bare Crossref title is maximally dense by construction —
   every token it has is a query token. A record called simply *"Differential privacy"*
   ranked **first** for "federated learning differential privacy medical imaging", with
   the highest cosine in the candidate set, above a dozen papers that actually do
   federated DP on medical images. See *The evidence prior* below.

### The evidence prior, and why the 8-query set could not have found it

Controlling for relevance grade, title-only records were ranking **15–31 percentile
points above equally-relevant records that had abstracts — at every grade**:

| grade | has abstract | title only | gap |
|-------|--------------|------------|-----|
| 3 | 62.6% (n=169) | 77.4% (n=45) | +14.8% |
| 2 | 36.1% (n=73)  | 66.7% (n=16) | +30.6% |
| 1 | 29.9% (n=51)  | 54.4% (n=9)  | +24.5% |
| 0 | 14.7% (n=41)  | 30.8% (n=7)  | +16.2% |

Same bias at every grade means a scoring artefact, not a quality difference. **The
pooled means hid it completely**: title-only records are *more* relevant on average
(mean grade 2.29 vs 2.11), because Crossref's stubs are often reviews. The first
measurement said "title-only records rank higher and deserve to" — and only controlling
for grade showed that was a coincidence covering a bug.

The fix is a document prior, `L / (L + k)` on token count, applied after fusion. It is
deliberately *not* a rule about missing abstracts: a two-sentence abstract carries more
evidence than a bare title and less than a full one, and a continuous weight can say so
where a boolean cannot.

**`k` was chosen by a paired bootstrap over the 20 queries, not by taking the best cell
of a sweep** — and that distinction changed the answer:

| k  | ΔNDCG@10 | 95% CI | queries better/worse |
|----|----------|--------|----------------------|
| 1  | +0.002 | [−0.002, +0.005] | 5 / 3 |
| **3** | **+0.014** | **[+0.004, +0.025]** | **10 / 3** |
| 8  | +0.006 | [−0.012, +0.026] | 8 / 6 |
| 20 | +0.005 | [−0.021, +0.035] | 8 / 6 |

Only `k ∈ [2, 4]` improves with an interval that excludes zero. The *largest* apparent
gain in the whole sweep was NDCG@5 **+0.038** at `k=20` — which looks like the winner
until the interval comes back `[−0.011, +0.105]` on a 6–4 split. Picking that number
would have been picking noise. The shipped effect is small, real, and reported as
small: **NDCG@10 0.901 → 0.915, MRR 0.950 → 0.975**, with `federated-learning-privacy`
going 0.651 → 0.717 and its MRR 0.500 → 1.000.

The ablation ships as a row in the evaluation script, so the prior stays falsifiable.

### A measured idea that did not pay off

BM25 is a bag of words, so "prime editing" is scored as two independent terms and a
paper matching *editing* + *efficiency* + *CRISPR* + *human* can outrank one actually
about prime editing. Indexing adjacent token pairs fixes exactly that query
(NDCG@10 **0.592 → 0.689**) but **lowers the average** (0.915 → 0.898), because doubling
the term space dilutes unigram IDF everywhere else. It ships off by default, with the
trade-off recorded in [`lexical.py`](backend/app/services/ranking/lexical.py) rather
than quietly dropped.

Re-measured on 20 queries, the conclusion survived the larger set — and `prime-editing`
is still the worst query on the board at NDCG@10 0.592, for exactly this reason. It is
the one place where the shipped default is knowingly the weaker choice for a specific
query in exchange for the average.

### Known limitations

- **Twenty queries is still a small set.** It is enough to bootstrap a confidence
  interval, which is why every claimed improvement above carries one, but differences
  under ~0.01 remain untrustworthy and a 95% interval on 20 paired samples is wide.
- **BM25's IDF is local to the candidate set.** For a query like "prime editing" where
  every candidate mentions prime editing, the phrase carries almost no lexical weight.
  On a two-candidate set the lexical signal collapses to zero entirely (Okapi IDF is 0
  at df = N/2); the hybrid correctly falls back to semantic there.
- **Judgments are single-annotator**, assigned by reading each title and abstract
  against a documented rubric. There is no inter-annotator agreement figure.
- **Papers with no abstract rank on title and keywords alone**, which is genuinely
  weaker evidence — a known cost of Crossref's incomplete abstract coverage. The
  evidence prior corrects the *scoring* bias this caused; it cannot invent the missing
  text, and such records are still ranked on less information than their neighbours.
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

## Stage 3 — entities and topic clusters

Two enrichment passes run over the ranked set, concurrently, in worker threads.

### NER, and being honest about a general-purpose model

The brief asks for spaCy "or a scientific-domain model like SciSpacy if available".
SciSpacy is not installed here, and the difference is not cosmetic. Run
`en_core_web_sm` — trained on news — over a real abstract and it returns:

| span | label | verdict |
|---|---|---|
| `pegRNA improves CRISPR-Cas9` | `LAW` | a technique read as legislation |
| `HEK293` | `GPE` | a cell line read as a country |
| `CRISPR-Cas9`, `FAB-CRISPR`, `HDR` | `ORG` | assays read as companies |
| `Transformer` | `ORG` | an architecture read as a company |
| `the Broad Institute` | `ORG` | correct |

So the extractor runs **two passes** and scopes how far it trusts each:

1. **Model NER**, with labels normalized onto one `EntityLabel` vocabulary so the API
   contract does not change when the model does. If a SciSpacy model *is* installed it
   is preferred and its labels are trusted in full. A general model is trusted only for
   `ORG` and `PERSON`, and only past three guards, each written after an observed
   failure above: reject spans shaped like technical terms, reject everything from the
   title (title case reads as proper nouns to a news model), and reject single-word
   `ORG`s without an institutional suffix.
2. **Pattern extraction** for the symbol-shaped terms a news model has never seen as a
   category — `pegRNA`, `CRISPR-Cas9`, `BRCA1`, `scRNA-seq`, `PE3`, `SARS-CoV-2`. These
   are what a researcher actually scans for. It only fills spans the first pass did not
   claim, so highlights never overlap.

After the guards, the same abstract yields `Broad Institute` and `Google Brain` as
organizations and `pegRNA`, `BERT`, `HEK293T`, `scRNA-seq` as technical terms.

Entities carry **every occurrence as character offsets into a named field**, so the
frontend can highlight inline in the title and abstract independently. A surface form
gets exactly one label — grouping by `(text, label)` previously let `PE` appear twice
in one paper, as a technical term and an organization, which is noise in a filter list
and a bug in a highlight layer.

### Clustering, and why silhouette alone is the wrong objective

Choosing *k* by maximizing silhouette is the obvious approach and it is wrong here. On
the 24-paper "diffusion models" pool, the best-scoring partition was **23 papers plus
one singleton, silhouette 0.481** — a near-perfect score for a split that gives a user
nothing to click. Average linkage on cosine distance produced that same singleton
chaining on every query tested.

Two changes, both measured:

- **Ward linkage** instead of average. It minimizes within-cluster variance and
  produced balanced partitions instead (`[15, 7]`, `[18, 6]`, `[16, 4, 3]`). Ward needs
  Euclidean distance, which is legitimate because the embedder returns L2-normalized
  vectors — squared Euclidean is then `2(1 − cosine)`, the same geometry.
- **Usability constraints** that silhouette does not measure: no cluster below 3
  papers, and no cluster holding more than 80% of the set. When no *k* satisfies both,
  the honest answer is that the result set is one coherent topic, and that is what gets
  reported.

Cluster names come from **c-TF-IDF** — term frequency within a cluster, discounted by
how many clusters contain the term, times the share of the term's occurrences falling
in that cluster. The exclusivity factor was a measured addition: without it the smaller
clusters were named in the parent topic's vocabulary (`alphafold · protein ·
prediction`); with it they name themselves (`alphafold · alphafold-multimer ·
differentiable`).

### What it does on the golden set

`python -m scripts.evaluate_ranking --clusters`

```
prime-editing          n=24  k=2  sizes=18/6     silhouette=0.4205
    [18] prime · editing · cells
    [ 6] irradiation · solar · critical        <- the off-topic papers, isolated
transformer-attention  n=23  k=3  sizes=16/4/3   silhouette=0.1995
    [16] models · attention · language
    [ 4] segmentation · medical · boundaries
    [ 3] molecular · clms · risk
protein-folding        n=22  k=2  sizes=15/7     silhouette=0.1282
    [15] protein · structure · prediction
    [ 7] alphafold · alphafold-multimer · differentiable
gnn-molecular          n=23  not split — one coherent topic
diffusion-image        n=24  not split — one coherent topic
```

**5 of 8 pools split into sub-topics; 3 correctly declined.** This is reported
descriptively rather than scored: nobody hand-labelled the "correct" sub-topics for
these queries, so there is no ground truth to compute an accuracy against. Publishing
one anyway would be a worse claim than publishing none. The silhouette scores, group
sizes and generated labels are printed so a reader can judge the output themselves.

### Both stages degrade like the rest of the pipeline

`EnrichmentReport` and `ClusteringReport` mirror `SourceReport` and `RankingReport`. If
no spaCy model can be loaded, extraction falls back to pattern-only and says so in
`entities.model`. If a result set is too small to cluster, or refuses to split, that is
a reported state with a reason — not an empty list the UI has to guess about.

### One embedding pass, not two

Ranking and clustering both embed the same papers. A `CachedEmbedder` wraps the model
with an LRU keyed by a digest of the text, so the second consumer reads its vectors out
of the cache instead of paying for another forward pass — roughly two seconds saved per
search on a 24-paper set. Misses are still encoded as one batch, because batching is
most of a transformer's CPU throughput. The cache outlives a single request too, so a
paper appearing in two refinements of a query is embedded once.

### Known limitations

- **No SciSpacy.** Its models are the right tool and would replace the pattern pass
  with typed diseases, chemicals and genes. The code already prefers them if installed;
  the label map and the trust switch are in place.
- **Residual ORG false positives.** The guards remove the systematic failures, not
  every one.
- **Clustering has no ground truth here** — see above.
- **Cluster labels are bag-of-words**, so they read as term lists rather than phrases.

---

## Stage 4 — grounded summarization

Every paper gets a 2-3 sentence summary. The interesting part is not generating it.

### The policy

```
cache hit?  ──yes──▶ return it
    │no
    ▼
generate ──▶ grounding check ──pass──▶ return, cache it
                    │fail
                    ▼
         regenerate, quoting the specific failures back
                    │
              ┌─────┴─────┐
            pass         fail
              │            │
              ▼            ▼
          return    extractive fallback, labelled
```

A summary that cannot be grounded is **replaced, not shown with a warning**. A
research tool should not offer "here is a claim we know the abstract does not
support" as an option. The fallback is sentences lifted verbatim from the abstract:
worse writing, but it cannot hallucinate, because it *is* the abstract.

The retry is a correction, not a re-roll. It quotes the rejected text and the exact
reasons back to the model — re-running the same prompt at a higher temperature would
just be sampling for luck.

### The grounding check is deliberately not an LLM

Asking a model to grade its own output is circular, doubles cost and latency, and
produces a verdict that cannot be unit-tested. These six checks are deterministic,
run in microseconds, and each is a property that can be asserted:

| check | catches |
|---|---|
| **Fabricated numbers** | every figure in the summary must occur in the abstract |
| **Fabricated entities** | gene symbols, acronyms, named methods the abstract never mentions |
| **Vocabulary overlap** | a summary written about a different paper |
| **Direction of effect** | "increased" silently becoming "decreased" |
| **Overclaiming** | "the first", "proves", "cures" — when the abstract doesn't say so |
| **Format** | the 2-3 sentences that were actually requested |

### The blind spot these six rules could not cover

Every rule above is lexical, and there is a class of fabrication that lexical rules
provably cannot see. A **recombination** is a fluent claim assembled entirely from the
abstract's own sentences that the abstract never actually makes — *"the editing
efficiency was caused by the pegRNA design"* invents no number, no entity, and no
reversed direction, and shares almost all its content words with the source.

Measured against 934 such cases, the six rules together caught **zero**. Not a low
rate — none.

The textbook answer is an NLI model, which means a second model to ship and one model
grading another. Instead the check reuses the embedder that is **already resident** for
ranking: every summary sentence should be a compression of something the abstract
says, so a sentence whose best alignment to any *contiguous window* of abstract
sentences is poor is asserting something the source does not. A recombination has high
vocabulary overlap and low sentence-level alignment, because it welds together claims
from sentences that never touch.

Two design decisions came out of measurement rather than taste:

**Windows, not single sentences.** Summarizing is compression — one good summary
sentence routinely condenses two or three consecutive abstract sentences — and scoring
against each individually **rejected 44% of legitimate summaries**.

**Advisory, not blocking.** The threshold trades detection against false alarms, and
the synthetic benchmark was badly misleading about the price:

| threshold | recombinations caught | false alarms on *synthetic* paraphrase | cautions on *live* generation |
|---|---|---|---|
| **0.50** | **61.3%** | 0.0% | **6.4%** |
| 0.55 | 71.3% | 0.2% | 10.6% |
| 0.65 | 87.2% | 0.5% | 29.8% |
| 0.70 | 92.6% | 0.6% | **43.6%** |
| 0.75 | 97.6% | 1.1% | — |

**Read the last two columns against each other.** They measure the same thing — how
often the check cries wolf — and they disagree by two orders of magnitude. On hand-built
paraphrases 0.70 costs 0.6% and looks nearly free, which is exactly what the synthetic
benchmark on its own would have recommended. Against real generated text the same
threshold cautions **43.6%**. The paraphrases were built by rewriting sentences; a model
summarizing an abstract compresses three sentences into one, and no rewrite rule does
that.

**A warning that fires on two summaries in five is one readers learn to skip**, which
would cost the signal entirely. So 0.50 ships — chosen on the live column, not the
synthetic one — support failures are shown as a caution with the summary standing, and
only the deterministic rules reject.

### Measured, including where it fails

`python -m scripts.evaluate_grounding` — **332 real abstracts, 2618 labelled cases**,
fully offline and deterministic.

```
should be ACCEPTED
  faithful                  331/332    99.7%
  paraphrase                328/332    98.8%

should be REJECTED (one rule each)
  fabricated number         114/114   100.0%
  fabricated entity         332/332   100.0%
  reversed direction         48/48    100.0%
  overclaim                 194/194   100.0%
  wrong paper               332/332   100.0%

recombination — built from the abstract's own sentences,
so every lexical rule passes them
  conflated_finding         111/194    57.2%
  invented_causation        118/182    64.8%
  invented_comparison       111/182    61.0%
  scope_inflation           119/194    61.3%
  swapped_roles             114/182    62.6%

recall on designed cases   1020/1020  100.0%
false-positive rate           5/664     0.8%
recombination caught        573/934    61.3%
```

**Read the last row, not the first five.** Each corruption in the middle block is a
clean instance of exactly the failure mode one rule was written to catch, so 100%
there confirms the rules fire — it is not evidence that real hallucinations get caught.
The rows that carry information:

- **False-positive rate: 0.8%** (5 of 664), measured on paraphrases reworded away from
  the abstract's exact sentences. That is the real cost — a checker that rejects good
  summaries isn't "safe", it just degrades everything to extractive text.
- **Recombinations: 0% → 61.3%.** Still the weakest row, and still the honest headline.
  The check is a similarity threshold, not entailment: it cannot separate "A causes B"
  from "B causes A" when both sentences discuss A and B together. Roughly two in five
  recombinations still get through.

The attacks that produce that number are generated from **each abstract's own
sentences** and live next to the checker in `attacks.py`, because a detection rate is
only worth something if the attacks are honestly hard. An earlier formulaic straw-man
version was replaced for exactly that reason.

### Measured against live generation

`python -m scripts.evaluate_summaries --limit 100 --quality` — **100 real abstracts**
spanning every domain in the pool, caching bypassed, `ministral-8b-latest`.

| outcome | n | share |
|---|---|---|
| grounded on first attempt | 78 | **78%** |
| rescued by the correcting retry | 16 | 16% |
| fell back to extractive | 6 | **6%** |
| **generated text accepted** | 94 | **94%** |

2.37 s per paper; 6 of the 100 accepted summaries carry a support caution.

This replaces an earlier 20-paper run that reported 100% acceptance and a 0% fallback
rate. **That number did not survive a five-fold larger sample** — at n=100 the retry
rescues most failures but not all, and six papers reach the extractive fallback. The
smaller figure is the trustworthy one.

What the first attempts were rejected *for* is the actionable part — a pass rate says
a model failed, this says how:

```
fabricated_entity        11     a method name the abstract never mentions
overclaim                 8     "the first", "proves", where the abstract doesn't
low_overlap               7     drift toward a different paper
unsupported_claim         3     a sentence the abstract does not back (advisory)
fabricated_number         2     a figure that appears nowhere in the source
```

### Two thresholds and a model, all tuned by measurement

**The overlap threshold was too strict.** It shipped at 0.55, a guess. Sweeping it
against the labelled set showed 0.45 is the *lowest* value that still detects 100% of
wrong-paper drift (0.40 drops to 99.3%, 0.30 to 97.3%). Re-running live generation at
both:

| threshold | first attempt | fell back | drift caught |
|---|---|---|---|
| 0.55 | 50% | 15% | 100% |
| **0.45** | **80%** | **5%** | **100%** |

The stricter value was rejecting genuine paraphrase — nine of ten first-attempt
rejections were `low_overlap` — which cost a retry each and pushed 15% of papers to
extractive text for no gain in safety whatsoever.

**Model choice is not driven by the grounding numbers.** All three viable models
(`ministral-3b/8b/14b-latest`) accepted 100% of generated text with 0% fallback, and
first-attempt rates of 75/70/80% are inside the noise at n=20. (That comparison was run
at n=20 and has not been repeated at n=100, where the shipped model's acceptance turned
out to be 94% rather than 100% — so treat it as "no model separated itself on a small
sample", not as a current measurement of any of the three.) The default is
`ministral-8b-latest` for its rate limit — 188 req/min covers a full result set, where
14b's 30 req/min would throttle a 24-paper search.

**Mistral allocates quota per model, not per account.** Worth stating because it cost
an hour: a valid key returned HTTP 429 with `x-ratelimit-limit-req-minute: 0` on
`mistral-small-latest` while `/v1/models` authenticated fine. Testing all 18
chat-capable models found 12 with real allowance (30–750 req/min) and 4 at zero. The
provider now raises a distinct `LLMQuotaError` for a zero allowance instead of retrying
it three times and calling it throttling.

**Models emit markdown even when told not to.** The first live run produced `replaces
**CRISPR-Cas9** with the smaller **Cas12a**`, which a web UI renders as literal
asterisks. The prompt now forbids it *and* the provider strips it, because a prompt is
a request rather than a guarantee — and the alternative, rendering model output as
markdown in the frontend, is an injection surface. That prompt change bumped
`PROMPT_VERSION` to 2, which is exactly what the versioned cache key exists for: every
summary written under v1 was invalidated rather than served.

### Caching, and the cache key

Summaries are the only expensive, non-deterministic and *chargeable* thing the
pipeline produces, so they persist in SQLite. The key is not the paper id but
`(paper_id, model, prompt_version, source_fingerprint)`:

- **model** — different model, different answer.
- **prompt_version** — after the prompt is tightened, text produced under the old
  instructions must not be served. Otherwise the cache silently undoes the fix.
- **source_fingerprint** — a digest of the exact title and abstract summarized.
  Deduplication merges field-wise, so a paper with no abstract on one search can have
  one on the next; the old summary described different input.

One subtlety worth naming: when generation fails and falls back, the text is filed
under the *configured model*, not under `"extractive"`. Keying it off the produced
text would guarantee a miss on every future lookup and recompute the fallback forever.
It still *reports* as extractive so the UI never mislabels it as AI-generated.

### It runs without an API key

`build_provider` returns `None` when no key is set, and every summary becomes
extractive, clearly labelled `SummaryOrigin.EXTRACTIVE`. A portfolio project that
can't be cloned and run without paid credentials is a worse project. Set
`PAPERPILOT_MISTRAL_API_KEY` to switch generation on; nothing else changes.

The demo marks the provenance of every line: `[AI|grounded]` for a first-attempt pass,
`[AI*|grounded]` for one the retry corrected, `[EXT|unverifiable]` for a paper with no
abstract to check against.

Provider errors, rate limits and timeouts all degrade the same way: a provider outage
costs the user their summaries, not their search results.

### Is the summary any *good*, though?

Everything above answers "is this false". None of it answers "is this useful" —
*"This paper studies proteins."* is perfectly grounded and every check passes it.

There are no human reference summaries here, and writing a few hundred would encode one
annotator's taste, so the metrics are intrinsic and **each is reported beside the same
metric for two baselines on the same abstracts**. The baselines *are* the measurement:
a coverage of 0.658 means nothing until lead-3 scores 0.721 on the same papers.

`python -m scripts.evaluate_summaries --limit 100 --quality`, 94 papers:

| metric | generated | lead-3 | extractive | reading |
|---|---|---|---|---|
| **coverage** | 0.658 | 0.721 | **0.730** | how much of the abstract survived |
| **compression** | 0.428 | 0.425 | 0.448 | summary words ÷ abstract words |
| **novelty** | **0.706** | 0.000 | 0.019 | share of bigrams not in the source |
| **longest copied span** | **0.081** | 1.000 | 0.590 | longest verbatim run |
| **lead bias** | **0.601** | 0.176 | 0.479 | where in the abstract it drew from |
| **redundancy** | **0.556** | 0.574 | 0.593 | most similar pair of its own sentences |

**The honest read: this is a trade, not a win.** The generated summary genuinely
rewrites rather than copies (novelty 0.706 against the extractive baseline's 0.019, and
it lifts no clause longer than 8% of itself) and it reads the *whole* abstract rather
than its opening (lead bias 0.601 against lead-3's 0.176). It also carries **about 10%
less of the abstract's content than simply taking three sentences**, at the same
length. If all you want is coverage, the fallback that ships for free is better.

Coverage is deliberately the mirror of the support check: support asks whether
everything in the summary came from the abstract (precision), coverage asks how much of
the abstract survived into the summary (recall). Together they bracket the two ways a
summary fails — inventing and omitting.

**A prompt fix aimed at the gap, which failed.** v2 asked only for "what was done and
what was found", so a v3 asked for the question, the method *and* the finding, plus a
rule against stopping after the background. Re-measured on 100 fresh papers it moved
coverage by **nothing at all** — 0.658 to 0.658 — while compression rose 0.428 → 0.464
and redundancy 0.556 → 0.597. The summaries got ~8% longer and carried exactly as much;
lead bias got *worse* (0.601 → 0.569), so the added rule did not even buy the thing it
named. v3 was reverted and the reasoning kept in `prompts.py` so it is not retried. The
coverage gap is not a prompt problem — it is what a 2–3 sentence budget costs.

### Known limitations

- **Lexical rules are still blind to two in five recombinations** — quantified above.
  This is the big one, and the semantic check narrowed it rather than closing it.
- **The corruption set is synthetic.** Real models do not hallucinate by uniformly
  resampling digits; recall on designed cases is an upper bound.
- **Single provider implemented.** The `LLMProvider` protocol is one method, so
  adding OpenAI or a local model is a ~40-line adapter, but only Mistral is written.
- **n=100 for the live numbers**, one model, one sampling temperature. Enough to settle
  the thresholds and to have overturned the earlier n=20 claim of 100% acceptance; not
  enough to separate two models whose first-attempt rates differ by five points.
- **The quality metrics are proxies, and reference-free.** A summary can score well on
  all six and still be a bad summary; a genuinely excellent terse summary will score
  low on coverage. They are reported as a comparison against baselines precisely
  because the absolute values are not meaningful. None of them measure factual
  correctness — that is what the grounding layer is for.

---

## Stage 5 — citation export

BibTeX, RIS, and plain text in APA 7th or Vancouver, for one paper or a bulk
selection. The brief asked for each format's *actual* spec rather than an
approximation, so the tests do not check strings we wrote — they feed our output to
**independent parsers** (`bibtexparser`, `rispy`) and assert that another
implementation reads back what we meant. Anything only we can parse is not really
BibTeX.

That decision paid for itself immediately: it found two real bugs before the code
was ever committed.

### The two bugs round-tripping caught

**Escaping corrupted its own output.** `escape()` replaced LaTeX control characters
in sequence, so `\` became `\textbackslash{}` — and the *later* brace rules then
escaped those braces into `\textbackslash\{\}`. Every backslash in a title came out
mangled. Chained `str.replace` can never be right here; the fix is a single regex
pass that touches each source character exactly once and never re-scans a
replacement.

**Author names were escaped twice.** `_authors()` escapes each name and then adds
*structural* braces around unsplittable ones — `{The Genome Editing Consortium}`,
which is how BibTeX is told "this is one name, not `Consortium, The Genome Editing`".
The generic field path then escaped that again into `\{The Genome…\}`, so BibTeX
stopped seeing a grouped name and started seeing punctuation. Ampersands in author
names were mangled the same way.

Neither is visible by reading the output. Both are obvious the moment a real parser
reads it.

### What "actually correct" means per format

**BibTeX**

| concern | handling |
|---|---|
| `& % $ # _ { } ~ ^ \` | escaped in one pass — an unescaped `&` breaks the *user's* build |
| Title lowercasing | `{CRISPR-Cas9}` brace-protected, or styles typeset it "Crispr-cas9" |
| Punctuation in protection | `{DNA}.` not `{DNA.}` — braces protect capitals, not full stops |
| Key collisions | `chen_prime_2022`, then `…2022a` — same group, same year is the normal case |
| Non-ASCII keys | `Zöller` → `zoller`, folded |
| Entry types | a preprint is `@misc`, a chapter `@incollection` — derived, not defaulted |
| Page ranges | `1021--1030`, BibTeX's double dash |
| `month` | an unquoted macro (`month = jul`), which is style-aware, not a literal |

**RIS**

| concern | handling |
|---|---|
| Tag grammar | exactly `TY  - JOUR` — one space instead of two and EndNote drops the field |
| Record bounds | `TY` first, `ER  - ` last |
| Line endings | CRLF, as the spec requires |
| Repeatable tags | one `AU  - ` line per author, not a joined string |
| Page ranges | split into `SP`/`EP`, including en-dashed ranges publishers really deposit |
| Newlines in values | collapsed, or they would be read as new tag lines |

**Plain text.** APA 7th and Vancouver, because the audience is split — APA is the
science default, Vancouver is what biomedical journals want, and this searches
PubMed. Both have exact author rules that are the usual source of wrong output: APA
lists up to 20 authors and for 21+ gives the first 19, an ellipsis, then the **final**
author (not the twentieth — the common bug); Vancouver lists 6 then `et al`, with no
periods between surname and initials.

### Papers are stored server-side, not posted back

Export needs the full record, and a search response is not something the server keeps.
The alternative — having the client POST the papers back — means trusting a caller's
copy of a record to generate a citation, so anyone could get a plausible-looking
citation for a paper that does not exist. Instead every search upserts its **enriched**
papers into SQLite (`PaperStore`), and export takes ids.

Papers are stored as their serialized model rather than shredded into columns. The
`Paper` model *is* the schema, it changes as stages are added, and nothing here queries
by field — search is the query engine, this is a keyed store.

A selection where some ids have aged out still exports the rest and reports the gap in
an `X-PaperPilot-Missing` header, rather than failing the whole request.

### Endpoints

| method | path | purpose |
|---|---|---|
| `POST` | `/api/export` | bulk export of a selection, capped at 500 ids |
| `GET` | `/api/export/{paper_id}?format=ris` | single paper — a GET so the UI can use a plain link |
| `GET` | `/api/export/formats/available` | so the format picker isn't hardcoded in the frontend |

Both return a download with `Content-Disposition`, and caller-supplied filenames are
stripped of quotes and path separators before they reach that header.

Try it without the API: `python -m scripts.demo_search "prime editing" --export ris`

---

## Stage 6 — the frontend

React 19, TypeScript in **full strict mode**, Tailwind. The parts worth arguing about:

### Types are generated from the API, not written next to it

`scripts/dump_openapi.py` dumps the FastAPI schema without starting a server, and
`openapi-typescript` turns it into `src/types/api.ts`. Hand-written interfaces
mirroring Pydantic models drift the moment a field is added, and drift *silently*.
Generated ones turn a backend rename into a compile error.

```bash
cd backend && python -m scripts.dump_openapi
cd ../frontend && npx openapi-typescript openapi.json -o src/types/api.ts
```

**This immediately found a backend bug.** Strict TypeScript reported
`paper.entities` as `Entity[] | undefined`, because Pydantic marks any field with a
default as *optional* in the JSON schema — reasonable for a request, wrong for a
response, since the server always serializes those fields. The schema was
under-describing what the API actually sends, and the cost was ~30 null checks in the
UI for values that can never be absent. Fixed at the source with
`json_schema_serialization_defaults_required` on a shared `ApiModel` base
([`app/models/base.py`](backend/app/models/base.py)), not papered over in the client.

### Strict mode means strict

Beyond `strict: true`, this enables `noUncheckedIndexedAccess` and
`exactOptionalPropertyTypes`. Those are the two that actually catch things here: the
API is full of genuinely nullable fields, and they are what stops
`paper.score.combined` from compiling when `score` may be null.

**A config bug that never failed anything.** `tsconfig.app.json` and
`tsconfig.node.json` are referenced projects, and neither set `composite: true`.
Without it, build mode does not know they are check-only: it computes their expected
outputs as emitted JavaScript, looks for `src/App.js`, never finds it because `noEmit`
is set, and rebuilds from scratch every single time. The `tsBuildInfoFile` both
configs specify was written and then never trusted.

```
$ tsc -b --verbose        # before
Project 'tsconfig.app.json' is out of date because output file 'src/App.js' does not exist

$ tsc -b --verbose        # after
Project 'tsconfig.app.json' is up to date because newest input 'src/App.tsx' is older
than output '.../tsconfig.app.tsbuildinfo'
```

Nothing ever failed, which is why it survived — the only symptom was every build
paying full cost. `isolatedModules` was missing too: Vite transpiles with esbuild one
file at a time, so anything needing whole-program knowledge to erase is a runtime bug
waiting to happen.

### Every pipeline stage is visible

The backend reports per-stage outcomes — sources, ranking, clustering, entities,
summaries — and [`PipelineStatus`](frontend/src/components/PipelineStatus.tsx) is the
component those reports were designed for. One quiet line when everything worked,
expandable to per-stage detail, and an amber banner naming the specific source that
failed. *"39 papers · 1 duplicate merged · arXiv unavailable"* is a real, useful
state, not an error.

Summaries carry their provenance for the same reason: `AI`, `AI · corrected` and
`From abstract` are visually distinct, and the detail view shows **why a first attempt
was rejected** — *"the summary states '12', which does not appear in the abstract"*.
Rendering all three identically would throw away everything Stage 4 does.

### The design system is tokens, and the palette is audited

Colour is declared once as CSS custom properties and mapped into Tailwind by semantic
name — `surface`/`sunken`, `strong`/`body`/`muted`/`faint`, `line`/`control`, `accent`.
Components say `bg-surface`, not `bg-white dark:bg-slate-900`. The alternative puts a
`dark:` variant on every colour utility in the app, and one forgotten variant is a
panel that is white-on-white for half the users. That happened twice here before the
tokens existed, and neither instance was visible without opening a dark screenshot.

A palette is the one part of a redesign that can be checked rather than argued about,
so [`scripts/contrast.mjs`](frontend/scripts/contrast.mjs) reads the real values out of
`index.css` and scores every shipped pairing against WCAG 2.2 in both themes. It gates
`npm run build`.

```
$ npm run contrast
=== light ===
  ok  10.95:1 (needs 4.5)  text-body on surface — summary text
  ok   5.20:1 (needs 4.5)  text-muted on surface — authors and metadata
  ok   3.25:1 (needs 3)    control on surface — input and checkbox borders
  …
All 48 pairings meet WCAG 2.2 AA.
```

It found three genuine failures the first time it ran: muted text at 4.46:1 on a
recessed panel, and control borders at 1.5:1 where 1.4.11 asks for 3. The second
produced a separate `--control` token, because a border that *identifies a control*
owes 3:1 while a hairline separating two rows does not — and holding every divider to
3:1 would put the whole page in cages.

### The accessibility bug that a screenshot cannot show

Every result card used to be `role="button"` with `tabIndex={0}`, wrapping a real link
*and* a real checkbox. It looked and behaved correctly. It is also invalid —
interactive elements must not nest — and it flattens the card for assistive
technology: the heading, the link to the publisher and the select control all collapse
into one announcement of *"button, open details for …"*.

The accessible version of "click anywhere" is a **stretched link**: the title is the
control, a transparent pseudo-element extends its hit area over the row, and the
genuinely interactive children sit above it. Same pointer behaviour, correct
semantics, one tab stop per control instead of two overlapping ones.

Alongside it: a skip link (2.4.1), which the app never had; focus rings as outlines
rather than ring-plus-offset, so a scroll container cannot clip them (2.4.11);
`prefers-reduced-motion` honoured; and provenance encoded in colour *and* text, never
colour alone (1.4.1).

### Verified by driving a real browser

[`scripts/capture.mjs`](frontend/scripts/capture.mjs) runs Playwright against both
servers and asserts each flow, failing loudly rather than producing a screenshot of
nothing. A frontend that typechecks and builds has proved nothing about whether it
renders.

```
=== light ===                            === dark ===
  ok  respects the light OS preference     ok  respects the dark OS preference
  ok  detects the query type               ok  3 sub-topic tabs rendered
  ok  renders 59 result cards              ok  filtering narrows 59 papers to 45
  ok  26 entities highlighted inline       ok  downloads paperpilot-2-references.bib
  ok  Escape closes the modal              ok  exactly the 2 selected papers
  ok  no console errors (0)                ok  no console errors (0)
=== mobile ===
  ok  results render at 390px              ok  no horizontal overflow

All checks passed.        # 42 assertions across light, dark and mobile
```

It also verifies the *downloaded file*, not just that a button was clickable: the
export must contain exactly the two selected papers as BibTeX entries.

The run also writes a screenshot gallery, which is how the four defects listed under
*Known limitations* below were found — none of them failed an assertion.

### On latency, honestly

A search takes **4–20 seconds**, and the search box is not "instant". Measured
breakdown, all three sources healthy vs. one hung:

```
pubmed      ok         1949ms      ranking        7ms
arxiv       timeout   12010ms      entities     751ms
crossref    ok         2913ms      clusters      32ms
                                   summaries   7663ms  (cold)
                                   summaries     51ms  (cached)
```

The fan-out is concurrent, so retrieval costs the *slowest* source, not their sum —
which means a hung source sets the floor. That is why the per-source budget was cut
from 20s to 12s during this stage: 12s is the measured worst case for PubMed's
two-call `esearch`/`efetch` pattern, so it is as low as it can go without cutting off
a healthy source. Summarization is the other real cost, and it is paid once per paper
thanks to the SQLite cache.

Streaming results as each source lands would fix the perceived wait properly. That is
an architectural change, not a polish item, and it is not done.

### Known limitations

- **No component tests.** The Playwright run is end-to-end verification, not a unit
  suite; a broken component fails the whole capture rather than one assertion.
- **The harness checks presence, not appearance.** It asserts that things render, are
  clickable and produce the right file — not that they *look* right. Four real
  defects shipped past a green run and were found by opening the screenshots: metadata
  separators that opened a wrapped line with a dangling middot, an export bar that
  covered the last two results, a cluster rail clipped mid-word with no sign it
  scrolled, and a detail modal that rendered **fully transparent** in dark mode. The
  last one is the sharpest example: `[role="dialog"]` was still visible to Playwright,
  just see-through. Asserting on computed background and geometry would close some of
  this gap; it is not done.
- **No result virtualisation.** 59 rows is fine; 500 would not be.
- **arXiv rate-limits by IP** and the block outlasts its documented 1-request-per-3-seconds
  window. Heavy development traffic will trip it, and the per-process politeness
  limiter resets on every restart — which is exactly how it got tripped here. The
  screenshots show the resulting degraded state, which is at least an honest demo of
  the feature designed for it.

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
| Crossref | Inline JATS (`<i>`, `<sub>`, `<scp>`) in **titles and journal names**, not just abstracts | Stripped at the boundary for every string field, not the two someone remembered |
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

## Packaging

`docker compose up --build` brings up the whole stack on
<http://localhost:5173> with no configuration.

### Two decisions that dominate the backend image

**CPU-only torch.** `sentence-transformers` pulls in PyTorch, and the
default wheel carries CUDA libraries this service will never touch —
nothing here uses a GPU. Installing from PyTorch's CPU index instead
downloads a **174 MB** wheel rather than roughly 2.5 GB.

**Models baked in at build time.** The embedding model (~90 MB) and the
spaCy pipeline (~12 MB) are downloaded during the build, not on first
request. Otherwise the first search after every deploy pays for a
download, and the container cannot start at all without reaching Hugging
Face — a bad property for something meant to be reproducible.

The frontend runtime image contains no Node at all: the build stage emits
static files and nginx serves them.

| image | size | compressed |
|---|---|---|
| `paperpilot-frontend` | 74 MB | 21 MB |
| `paperpilot-backend` | dominated by torch + models | — |

`npm run build` runs `tsc -b` first, so a type error fails the image build
rather than shipping.

### nginx, and why the app is same-origin in both environments

The frontend container proxies `/api` and `/health` to the backend over
the compose network, exactly as the Vite dev server proxies them in
development. CORS is therefore not load-bearing in either environment —
it is configured, but nothing depends on it being right to work locally.

Verified against a running container rather than by reading the config:

- SPA served, and a deep link falls back to `index.html` rather than a
  404 from nginx
- `/api` and `/health` proxied through to the backend
- `index.html` is `no-cache`; fingerprinted assets are
  `public, max-age=31536000, immutable`
- gzip negotiated on the JS bundle

That check caught a real bug: `expires 1y` **plus** `add_header
Cache-Control` emits *two* `Cache-Control` headers, and which one a cache
honours is up to the cache. It is one directive now.

The full browser suite in
[`frontend/scripts/capture.mjs`](frontend/scripts/capture.mjs) also runs
against the containerized production build, not just the dev server.

### What is not verified here

The backend image was **not built to completion on this machine.** The
build is correct as far as it ran — apt, the venv, and CPU-only torch
resolving to the right wheel — but downloads on this network measured
**~57 KB/s**, which puts the torch wheel alone at roughly 50 minutes and
the full dependency set well beyond that. The Dockerfile now sets
`PIP_RETRIES` and `PIP_DEFAULT_TIMEOUT` so a slow link retries rather than
failing a build twenty minutes in, but the finished image size is unmeasured
and the compose stack has not been exercised end to end. Treat the
frontend image and the nginx configuration as verified and the backend
image as reviewed but unbuilt.


## Testing

```bash
cd backend
python -m pytest          # 449 tests
python -m ruff check app tests
python -m mypy app scripts

cd ../frontend
npm run contrast          # WCAG audit of the palette, both themes
npm run verify            # drive a real browser, 42 assertions
```

Every push runs ruff, mypy and the full pytest suite, plus the frontend's
type-check, lint, contrast audit and build
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

CI also enforces the **generated-types chain**, which is the one thing a reviewer
cannot check by reading: the backend job regenerates `openapi.json` from the Pydantic
models and fails if the committed schema has drifted, and the frontend job
regenerates `src/types/api.ts` from that schema and fails if *it* has drifted.
Together they prove the TypeScript still describes the Python. Both halves were
tested by deliberately breaking them.

**The browser harness is deliberately not in CI.** It drives the real app against
PubMed, arXiv and Crossref, and arXiv rate-limits by IP — a CI runner is the most
shared address there is. A suite that fails for reasons unrelated to the change
teaches people to ignore it, so `npm run verify` stays a local command and this
limitation is stated rather than hidden behind a green badge.

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
| Embedders, BM25, the fusion strategies, and the evidence prior | `tests/test_ranking.py` |
| IR metrics, hand-computed from their definitions | `tests/test_ranking_evaluation.py` |
| Entity patterns, label normalization, trust guards | `tests/test_entities.py` |
| Cluster selection, labelling, and refusal to split | `tests/test_clustering.py` |
| Embedding cache correctness and eviction | `tests/test_embedding_cache.py` |
| Every grounding rule, and the limitation it cannot cover | `tests/test_grounding.py` |
| The semantic support check, and the blocking/advisory split | `tests/test_support.py` |
| Retry-on-failure, fallback, caching, provider errors | `tests/test_summarization.py` |
| Quality metrics, each against a case where its value is known | `tests/test_summary_quality.py` |
| BibTeX and RIS round-tripped through independent parsers | `tests/test_export.py` |
| Export endpoints, the paper store, filename sanitizing | `tests/test_export_api.py` |
| Palette contrast, both themes, gating the build | `frontend/scripts/contrast.mjs` |
| Every user flow, in a real browser | `frontend/scripts/capture.mjs` |

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness plus the registered sources (the frontend builds its filters from this) |
| `GET` | `/api/search?q=…` | Search all sources, ranked, with per-source and ranking status |
| `POST` | `/api/export` | Export a selection as BibTeX, RIS, APA or Vancouver |
| `GET` | `/api/export/{id}` | Export one paper |
| `POST` | `/api/search` | Same, for abstract snippets too long for a query string |
| `GET` | `/api/parse?q=…` | How a query would be interpreted — powers the live input-type hint in the UI |

Interactive docs at `/docs` when the server is running.

---

## Tech stack

**Backend** Python 3.12 · FastAPI · Pydantic v2 · httpx (async) · defusedxml
**Ranking** sentence-transformers (`all-MiniLM-L6-v2`) · rank-bm25 · NumPy
**Enrichment** spaCy (SciSpacy-ready) · scikit-learn (Ward agglomerative)
**Summarization** Mistral API via httpx · deterministic grounding check · SQLite cache
**Export** BibTeX · RIS · APA 7th · Vancouver, validated against independent parsers
**Frontend** React 19 · TypeScript (strict) · Tailwind (semantic tokens) · Vite · types generated from OpenAPI
**Testing** pytest · pytest-asyncio · respx · bibtexparser · rispy · ruff · mypy (strict) · Playwright · WCAG contrast audit
**Packaging** Docker · docker compose · nginx
