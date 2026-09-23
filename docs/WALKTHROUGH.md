# Demo walkthrough

A shot-by-shot script for a 4–5 minute screen recording. Written so the
recording can be made in one take, and so that whoever is watching sees the
engineering rather than a list of features.

The through-line: **every stage of this pipeline reports what it did, and
every quality claim is measured.** Say that at the start and return to it.

---

## Before recording

```bash
# 1. Start the stack (or the two dev servers — see the README quickstart)
docker compose up --build

# 2. Warm the caches so the demo is not waiting on cold summaries.
#    Run the exact queries you plan to demo, once.
curl -s -X POST http://localhost:5173/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"CRISPR prime editing efficiency in human cells","limit_per_source":12}' > /dev/null
```

- Clear the search history sidebar (**Clear**) so the first shot is empty.
- Set the OS theme to light — the dark-mode switch lands better as a reveal.
- Zoom the browser to ~110%; default text is small on a recording.
- Have a second query ready to paste: an abstract (below) and a DOI.

**Timing note.** A cold search takes 4–20 seconds. That is a real
constraint, not a bug, and the honest options are to warm the cache first
or to talk over the loading state — which is designed to be talked over,
since it names the three sources it is waiting on. Do not cut the wait out
and imply it is instant.

---

## 1. The problem (~30s)

> "A researcher tracking a topic searches PubMed, arXiv and Crossref
> separately. Three query syntaxes, three response formats, and then they
> skim dozens of abstracts by hand and format citations by hand."

Land on the empty state. Point at the search box hint: topic, keyword, DOI,
arXiv id, PMID, or a pasted abstract — **one box, four input types**.

## 2. One query, three sources (~60s)

Type `CRISPR prime editing efficiency in human cells`. **Pause before
pressing enter** and point at the `Topic` chip: the query type is detected
server-side as you type, so the hint cannot disagree with what the search
will do.

Press enter. While it runs, narrate the loading state — it names the three
sources it is fanning out to, concurrently.

When results land, point at the status bar:

> "39 papers, 1 duplicate merged, and it tells me arXiv was unavailable.
> That is not an error state — it is the search telling me the result set
> is incomplete and in which direction. A tool that quietly returned two
> sources' worth of results would be lying by omission."

Expand the status bar. Every stage reports: sources, ranking, clustering,
entities, summaries — with timings and models.

## 3. Ranking, and how it was measured (~45s)

Point at a relevance bar; hover for the breakdown.

> "Hybrid: embedding similarity plus BM25. And that is not a claim I am
> making by eye —"

Cut to the terminal:

```bash
cd backend && python -m scripts.evaluate_ranking
```

> "Twenty queries, 411 candidates, every one hand-judged. Hybrid gets
> Recall@10 of 0.630 against a ceiling of 0.669 — 94% of what any ranking
> could achieve, because most queries have more relevant papers than k."

Point at the last row of the table:

> "That row is the ablation. Growing the golden set from 8 queries to 20
> exposed a bug: records that are nothing but a title were out-ranking real
> papers, because both scorers reward term density and a two-word title is
> maximally dense. Controlling for relevance grade they ranked 15 to 31
> percentiles too high — at every grade, so it was scoring, not quality.
> The pooled averages hid it, because those stubs are genuinely more
> relevant on average. A document-length prior fixed it: NDCG@10 0.901 to
> 0.915."

The honest bit, worth saying out loud:

> "The k in that prior came from a paired bootstrap, not from the best cell
> of a sweep — and that mattered. The biggest number in the sweep was
> NDCG@5 +0.038, and its confidence interval spans zero on a 6-4 split.
> The one I shipped is +0.014 with an interval that excludes zero. I took
> the smaller, real effect."

## 4. Sub-topics (~30s)

Click a cluster tab. 39 papers → 23.

> "Ward linkage, k chosen by silhouette. Choosing k by silhouette alone is
> actually wrong — it prefers isolating a single outlier, which scores
> beautifully and gives you nothing to click. So there are two constraints
> silhouette does not measure: no cluster under three papers, and none
> holding more than 80% of the set. When nothing satisfies both, it says
> the results are one coherent topic instead of inventing tabs."

## 5. Grounded summaries — the centrepiece (~75s)

Point at the summary badges on the cards: `AI`, `AI · corrected`,
`From abstract`.

Open a paper that was **corrected**. Scroll to *First attempt rejected for*:

> "The model's first attempt said '12', which does not appear in the
> abstract, and mentioned Cas12a, which the abstract never names. So it was
> regenerated with those exact problems quoted back — a correction, not a
> re-roll at a higher temperature."

> "And the check is deliberately not another LLM call. Asking a model to
> grade its own output is circular, doubles the cost, and produces a verdict
> you cannot unit-test. These are six deterministic rules."

Cut to the terminal:

```bash
python -m scripts.evaluate_grounding
```

**Point at the last row, not the first five.** This is the most important
thirty seconds of the demo:

> "Recall on designed corruptions is 100%, and that number is nearly
> meaningless — each corruption is a clean instance of exactly the failure
> one rule was written to catch. The rows that matter are the last two: a
> 0.8% false-positive rate on paraphrased text, which is the real cost of
> checking, and the recombination row."

> "A recombination is a claim built entirely from the abstract's own
> sentences that the abstract never makes. The six lexical rules caught
> **zero of 934** — not a low rate, none. So there's a seventh check that
> compares each summary sentence against contiguous windows of the
> abstract, reusing the embedder that's already loaded for ranking rather
> than shipping an NLI model to have one model grade another. That takes it
> from 0% to 61%."

The number to be careful about, and the best thing to say here:

> "61% is the operating point, not the maximum. At a stricter threshold it
> catches 95% — and cautions **44% of perfectly good summaries**. A warning
> that fires on two summaries in five is one a reader learns to skip, so
> it would cost the signal entirely. At the shipped threshold it cautions 5
> in 100. And the synthetic benchmark lied about this: on hand-built
> paraphrases the strict threshold looked nearly free. Only live generation
> showed the real price, because summarizing compresses harder than any
> rewrite rule does."

> "Support failures are shown as a caution and the summary stands. Only the
> deterministic rules reject. Two in five recombinations still get through —
> it's a similarity threshold, not entailment."

### 5b. Is it any *good*, though? (~30s)

```bash
python -m scripts.evaluate_summaries --limit 100 --quality
```

> "Everything so far asks whether a summary is *false*. None of it asks
> whether it's *useful* — 'This paper studies proteins' is perfectly
> grounded. So this scores the generated summaries next to two baselines on
> the same abstracts: the first three sentences, and the extractive
> fallback that ships for free."

Point at the coverage row:

> "The generated summary genuinely rewrites — 71% of its bigrams aren't in
> the source, against 2% for the extractive baseline, and it never lifts a
> clause longer than 8% of itself. It reads the whole abstract rather than
> the opening. And it carries about 10% *less* of the abstract's content
> than just taking three sentences, at the same length. That's a trade, not
> a win, and I report it as one."

If asked whether that can be fixed:

> "I tried. I rewrote the prompt to ask for the question, the method and
> the finding explicitly. Coverage moved from 0.658 to 0.658 — nothing —
> while the summaries got 8% longer. I reverted it and left the reasoning
> in the file so I don't try it again. It isn't a prompt problem; it's what
> a three-sentence budget costs."

## 6. Entities (~20s)

Same modal, scroll to the abstract. Toggle **Highlight entities**.

> "Gene symbols, assays, acronyms — highlighted inline by character offset.
> A general spaCy model is trained on news and tags CRISPR-Cas9 as an
> organisation, so its labels are trusted only for institutions and people;
> the domain terms come from a shape-based pass."

## 7. Export (~40s)

**Select** → tick two papers → the export bar docks in. **BibTeX**. Open
the downloaded file.

> "No confirmation step. And the formats are spec-correct, not
> approximately correct — the tests round-trip the output through
> bibtexparser and rispy, so 'correct' means another implementation agrees."

> "That found two real bugs. Escaping corrupted its own output: a backslash
> became `\textbackslash{}`, then the brace rules escaped those braces. And
> author names were escaped twice, which broke the structural braces BibTeX
> uses to keep a consortium name together. Neither is visible by reading the
> output. Both are obvious the moment a parser reads it."

## 8. The other input types (~30s)

Paste this abstract:

> We present a method for correcting technical variation between batches of
> single-cell transcriptomic data. Our approach identifies mutual nearest
> neighbours between batches and uses them to estimate a correction vector,
> without requiring the populations to be shared across batches.

Point at the `Abstract` chip and the distilled keywords:

> "No API accepts a paragraph as a query, so the upstream calls get
> keywords — but the ranker embeds the whole paragraph. That is why the
> parser keeps both."

Then paste `10.1038/s41586-019-1711-4` → `Identifier`, direct lookup.

## 9. Close (~30s)

Toggle dark mode. Show the history sidebar. Narrow the window to phone width.

> "Types are generated from the backend's OpenAPI schema, so they cannot
> drift — that caught a real bug where the schema described response fields
> as optional that the server always sends. And the whole frontend is
> verified by driving a real browser: 42 assertions, including reading the
> bytes of the downloaded BibTeX."

```bash
cd frontend && node scripts/capture.mjs
```

---

## Do not skip these

They are what separates this from a feature tour:

1. **The degraded-source banner.** The design point of the whole reporting layer.
2. **"First attempt rejected for".** The grounding layer made visible.
3. **The recombination row, 0% → 61%.** Measuring where your own work fails is
   the most credible thing in the recording — and the honest version is that
   it went from "blind" to "misses two in five", not to "solved".
4. **The recall ceiling.** 0.630 reads like a bad number until you say the
   ceiling is 0.669.
5. **The quality table where the LLM loses.** Generated summaries carry ~10%
   *less* of the abstract than just taking three sentences. Say that out
   loud: it is the clearest signal that these numbers were measured rather
   than chosen.
6. **The bootstrap, not the sweep maximum.** The largest number in the sweep
   was noise; the shipped one is small and real.

## Do not claim

- That search is instant. It is 4–20 seconds, and the README says why.
- That hybrid ranking is decisively better than semantic alone. On this set
  it is not.
- That the grounding check catches hallucinations in general. It catches
  fabricated numbers, entities, reversed directions and drift, and about
  three in five recombinations. It does not do entailment.
- That the AI summary is better than the extractive fallback. It is more
  abstractive and reads more of the abstract; it also covers less of it.
  That trade is measured, and it is a trade.
