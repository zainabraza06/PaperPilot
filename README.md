# PaperPilot

AI-powered scientific literature search across **PubMed**, **arXiv** and **Crossref** —
one query, one ranked list, AI summaries, one-click citation export.

> **Status: Stages 1–4 complete — multi-source retrieval, hybrid ranking with
> measured Recall@k / NDCG@k, NER + topic clustering, and grounded AI
> summarization with a measured fact-checking layer.** Citation export
> (Stage 5), the React frontend (Stage 6) and packaging (Stage 7) are in progress.

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

# Compare ranking strategies on the golden set, and report clustering
python -m scripts.evaluate_ranking --per-query --sweep-alpha --clusters

# Measure the grounding check against 894 labelled cases
python -m scripts.evaluate_grounding

# Measure live generation (needs a Mistral key)
python -m scripts.evaluate_summaries --limit 20

# Or run the API
uvicorn app.main:app --reload
# → http://127.0.0.1:8000/docs
```

The first run downloads the embedding model (~90 MB) and needs a spaCy model
(`python -m spacy download en_core_web_sm`). Both load once at startup, in a worker
thread, so no search pays for them — and both degrade rather than fail if absent.

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

### Measured, including where it fails

`python -m scripts.evaluate_grounding` — 148 real abstracts from the frozen pool,
894 labelled cases, fully offline and deterministic.

```
should be ACCEPTED
  faithful                  148/148   100.0%
  paraphrase                146/148    98.6%

should be REJECTED (one rule each)
  fabricated number          47/47    100.0%
  fabricated entity         148/148   100.0%
  reversed direction         22/22    100.0%
  overclaim                  85/85    100.0%
  wrong paper               148/148   100.0%

known blind spot
  recombination               0/148     0.0%   ← 100% slip through
```

**Read the last row, not the first five.** Each corruption in the middle block is a
clean instance of exactly the failure mode one rule was written to catch, so 100%
there confirms the rules fire — it is not evidence that real hallucinations get
caught. The two rows that carry information:

- **False-positive rate: 0.7%** (2 of 296), measured on paraphrases reworded away from
  the abstract's exact sentences. That is the real cost of the check — a checker that
  rejects good summaries isn't "safe", it just degrades everything to extractive text.
- **The blind spot is 100%.** A "recombination" case asserts a causal link the abstract
  never makes, built entirely from the abstract's own vocabulary — no invented number,
  no invented entity, no reversed direction. Every single one passes. The check is
  **lexical, not inferential**, and catching these needs entailment, which is a model,
  which brings back every problem above. This is measured and stated rather than left
  for someone to discover.

There is a test (`test_the_check_is_documented_as_lexical_not_inferential`) that pins
this limitation, so a future change claiming to fix it has to update the test.

### Measured against live generation

`python -m scripts.evaluate_summaries --limit 20` — 20 real abstracts spanning every
domain in the pool, caching bypassed, `ministral-8b-latest`.

| outcome | n | share |
|---|---|---|
| grounded on first attempt | 14 | **70%** |
| rescued by the correcting retry | 6 | 30% |
| fell back to extractive | 0 | **0%** |
| **generated text accepted** | 20 | **100%** |

0.41 s per paper. The retry rescued every single failure, which is the strongest
evidence that quoting the specific issues back beats re-sampling: none of the six
needed a third attempt or a fallback.

What the first attempts were rejected *for* is the actionable part — a pass rate says
a model failed, this says how:

```
overclaim                 4     "the first", "proves", where the abstract doesn't
fabricated_entity         3     a method name the abstract never mentions
contradicted_direction    1     an effect stated in the wrong direction
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
first-attempt rates of 75/70/80% are inside the noise at n=20. The default is
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

### Known limitations

- **Lexical, not inferential** — quantified above. This is the big one.
- **The corruption set is synthetic.** Real models do not hallucinate by uniformly
  resampling digits; recall on designed cases is an upper bound.
- **Single provider implemented.** The `LLMProvider` protocol is one method, so
  adding OpenAI or a local model is a ~40-line adapter, but only Mistral is written.
- **n=20 for the live numbers.** Enough to show the retry is doing real work and to
  settle the threshold; not enough to separate three models whose first-attempt rates
  differ by five points.
- **Prose quality is unmeasured.** Everything above scores whether a summary is
  *grounded*, not whether it is *good*. A grounded summary can still be a bland
  restatement of the first sentence, and nothing here would catch that.

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
backend/app/
├── api/            # thin routes + dependency wiring
├── core/           # text normalization, errors, logging, rate limiting, safe XML
├── models/         # Paper, Author, and the search request/response contract
├── services/       # query parsing, deduplication, search orchestration
│   ├── enrichment/ # NER (spaCy + shape patterns) and topic clustering
│   ├── ranking/    # document view, embeddings, cache, BM25, fusion, IR metrics
│   └── summarization/  # providers, prompts, grounding check, extractive fallback
├── storage/        # SQLite summary cache
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
python -m pytest          # 316 tests
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
| Entity patterns, label normalization, trust guards | `tests/test_entities.py` |
| Cluster selection, labelling, and refusal to split | `tests/test_clustering.py` |
| Embedding cache correctness and eviction | `tests/test_embedding_cache.py` |
| Every grounding rule, and the limitation it cannot cover | `tests/test_grounding.py` |
| Retry-on-failure, fallback, caching, provider errors | `tests/test_summarization.py` |

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
**Enrichment** spaCy (SciSpacy-ready) · scikit-learn (Ward agglomerative)
**Summarization** Mistral API via httpx · deterministic grounding check · SQLite cache
**Testing** pytest · pytest-asyncio · respx · ruff · mypy (strict)
**Coming** citation export (Stage 5) · React + TypeScript + Tailwind (Stage 6) ·
Docker Compose (Stage 7)
