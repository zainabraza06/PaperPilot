"""Summarizer, provider and cache tests.

The policy under test is: never show a claim the abstract does not
support, and never let that policy cost the user their search results.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from app.models.paper import Paper, SourceName
from app.models.summary import GroundingStatus, Summary, SummaryOrigin
from app.services.summarization.extractive import ExtractiveSummarizer
from app.services.summarization.grounding import GroundingChecker
from app.services.summarization.prompts import PROMPT_VERSION
from app.services.summarization.provider import (
    LLMQuotaError,
    LLMRateLimitedError,
    LLMUnavailableError,
    MistralProvider,
    build_provider,
)
from app.services.summarization.summarizer import PaperSummarizer
from app.storage.summary_cache import SummaryCache, fingerprint

ABSTRACT = (
    "Prime editing enables precise genome edits without double-strand breaks. "
    "We benchmarked PE2 and PE3 across six primary human cell types using pegRNAs. "
    "Editing efficiency increased to 42% in T cells, with indel rates below 1.5%."
)
GOOD_SUMMARY = (
    "The authors benchmarked PE2 and PE3 across six primary human cell types. "
    "Editing efficiency increased to 42% in T cells, with indel rates below 1.5%."
)
BAD_SUMMARY = "This first-ever work proves efficiency reached 87% using CRISPRoff."


def make_paper(paper_id: str = "p1", abstract: str | None = ABSTRACT) -> Paper:
    return Paper(
        id=paper_id,
        source=SourceName.PUBMED,
        source_id="1",
        title="Prime editing efficiency in primary human cells",
        abstract=abstract,
        url="https://example.org/1",
    )


class ScriptedProvider:
    """Returns queued completions, recording the prompts it was given."""

    model_id = "scripted-model"

    def __init__(self, *responses: str | Exception) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    async def complete(self, system: str, user: str, *, max_tokens: int = 300) -> str:
        self.prompts.append(user)
        result = self._responses.pop(0) if self._responses else "fallback text."
        if isinstance(result, Exception):
            raise result
        return result


# --- the happy path -------------------------------------------------------


async def test_a_grounded_summary_is_returned_as_generated() -> None:
    summarizer = PaperSummarizer(ScriptedProvider(GOOD_SUMMARY))
    papers, report = await summarizer.summarize([make_paper()])

    summary = papers[0].summary
    assert summary is not None
    assert summary.text == GOOD_SUMMARY
    assert summary.origin is SummaryOrigin.GENERATED
    assert summary.grounding.status is GroundingStatus.GROUNDED
    assert summary.attempts == 1
    assert report.applied and report.summarized == 1 and report.fell_back == 0


async def test_every_paper_gets_a_summary_in_order() -> None:
    papers = [make_paper(f"p{n}") for n in range(3)]
    result, report = await PaperSummarizer(ScriptedProvider(*[GOOD_SUMMARY] * 3)).summarize(
        papers
    )
    assert [p.id for p in result] == ["p0", "p1", "p2"]
    assert all(p.summary is not None for p in result)
    assert report.summarized == 3


# --- correction and fallback ---------------------------------------------


async def test_an_ungrounded_summary_triggers_a_correcting_retry() -> None:
    provider = ScriptedProvider(BAD_SUMMARY, GOOD_SUMMARY)
    papers, report = await PaperSummarizer(provider).summarize([make_paper()])

    summary = papers[0].summary
    assert summary is not None
    assert summary.origin is SummaryOrigin.REGENERATED
    assert summary.attempts == 2
    assert report.regenerated == 1


async def test_the_retry_quotes_the_specific_failures_back() -> None:
    # A re-roll at a higher temperature would be sampling for luck; naming
    # the failures makes the second attempt a correction.
    provider = ScriptedProvider(BAD_SUMMARY, GOOD_SUMMARY)
    await PaperSummarizer(provider).summarize([make_paper()])

    retry_prompt = provider.prompts[1]
    assert BAD_SUMMARY in retry_prompt
    assert "87%" in retry_prompt
    assert "CRISPRoff" in retry_prompt


async def test_a_summary_that_cannot_be_grounded_is_replaced_not_shown() -> None:
    # Showing text known to be ungrounded, even with a warning, is not an
    # option a research tool should offer.
    provider = ScriptedProvider(BAD_SUMMARY, BAD_SUMMARY)
    papers, report = await PaperSummarizer(provider).summarize([make_paper()])

    summary = papers[0].summary
    assert summary is not None
    assert summary.origin is SummaryOrigin.EXTRACTIVE
    assert "87%" not in summary.text
    assert summary.text in ABSTRACT or all(
        sentence.strip() in ABSTRACT for sentence in summary.text.split(". ") if sentence
    )
    assert report.fell_back == 1


async def test_a_provider_outage_costs_summaries_not_search_results() -> None:
    provider = ScriptedProvider(LLMUnavailableError("HTTP 503"))
    papers, report = await PaperSummarizer(provider).summarize([make_paper()])

    assert papers[0].summary is not None
    assert papers[0].summary.origin is SummaryOrigin.EXTRACTIVE
    assert report.applied is True


async def test_with_no_provider_summaries_are_extractive_and_labelled() -> None:
    papers, report = await PaperSummarizer(None).summarize([make_paper()])

    summary = papers[0].summary
    assert summary is not None
    assert summary.origin is SummaryOrigin.EXTRACTIVE
    assert summary.is_ai_generated is False
    assert report.model == "extractive"
    assert "no LLM provider configured" in (report.reason or "")


async def test_a_paper_with_no_abstract_is_never_sent_to_the_model() -> None:
    # Nothing could ground the result, so generating would be spending money
    # on an unverifiable claim.
    provider = ScriptedProvider(GOOD_SUMMARY)
    papers, _ = await PaperSummarizer(provider).summarize([make_paper(abstract=None)])

    assert provider.prompts == []
    assert papers[0].summary is not None
    assert papers[0].summary.grounding.status is GroundingStatus.UNVERIFIABLE


async def test_summarizing_nothing_is_not_an_error() -> None:
    papers, report = await PaperSummarizer(None).summarize([])
    assert papers == []
    assert report.applied is False


# --- caching --------------------------------------------------------------


@pytest.fixture
def cache(tmp_path: Path) -> SummaryCache:
    return SummaryCache(tmp_path / "test.db")


async def test_a_cached_summary_is_not_regenerated(cache: SummaryCache) -> None:
    provider = ScriptedProvider(GOOD_SUMMARY, GOOD_SUMMARY)
    summarizer = PaperSummarizer(provider, cache=cache)

    await summarizer.summarize([make_paper()])
    papers, report = await summarizer.summarize([make_paper()])

    assert len(provider.prompts) == 1  # only the first search called the model
    assert papers[0].summary is not None and papers[0].summary.cached is True
    assert report.from_cache == 1


def test_the_cache_round_trips_a_summary_with_its_verdict(cache: SummaryCache) -> None:
    verdict = GroundingChecker().check(GOOD_SUMMARY, ABSTRACT)
    summary = Summary(
        text=GOOD_SUMMARY,
        origin=SummaryOrigin.GENERATED,
        grounding=verdict,
        model="m1",
        prompt_version=PROMPT_VERSION,
    )
    key = fingerprint("title", ABSTRACT)
    cache.put("p1", "m1", PROMPT_VERSION, key, summary)

    restored = cache.get("p1", "m1", PROMPT_VERSION, key)
    assert restored is not None
    assert restored.text == GOOD_SUMMARY
    assert restored.grounding.status is GroundingStatus.GROUNDED
    assert restored.cached is True


def test_a_different_model_does_not_hit_the_cache(cache: SummaryCache) -> None:
    key = fingerprint("title", ABSTRACT)
    summary = Summary(
        text=GOOD_SUMMARY,
        origin=SummaryOrigin.GENERATED,
        grounding=GroundingChecker().check(GOOD_SUMMARY, ABSTRACT),
        model="m1",
    )
    cache.put("p1", "m1", PROMPT_VERSION, key, summary)
    assert cache.get("p1", "m2", PROMPT_VERSION, key) is None


def test_a_prompt_change_invalidates_cached_text(cache: SummaryCache) -> None:
    # Otherwise tightening the prompt would be silently undone by the cache.
    key = fingerprint("title", ABSTRACT)
    summary = Summary(
        text=GOOD_SUMMARY,
        origin=SummaryOrigin.GENERATED,
        grounding=GroundingChecker().check(GOOD_SUMMARY, ABSTRACT),
        model="m1",
        prompt_version=1,
    )
    cache.put("p1", "m1", 1, key, summary)
    assert cache.get("p1", "m1", 2, key) is None


def test_a_changed_abstract_invalidates_cached_text(cache: SummaryCache) -> None:
    # Deduplication merges field-wise, so a paper can gain an abstract
    # between searches; its old summary described different input.
    summary = Summary(
        text=GOOD_SUMMARY,
        origin=SummaryOrigin.GENERATED,
        grounding=GroundingChecker().check(GOOD_SUMMARY, ABSTRACT),
        model="m1",
    )
    cache.put("p1", "m1", PROMPT_VERSION, fingerprint("title", None), summary)
    assert cache.get("p1", "m1", PROMPT_VERSION, fingerprint("title", ABSTRACT)) is None


async def test_an_extractive_fallback_is_cached_under_the_configured_model(
    cache: SummaryCache,
) -> None:
    # Filing it under "extractive" would guarantee a miss on every future
    # lookup, so the fallback would be recomputed forever.
    provider = ScriptedProvider(BAD_SUMMARY, BAD_SUMMARY)
    summarizer = PaperSummarizer(provider, cache=cache)
    await summarizer.summarize([make_paper()])

    restored = cache.get(
        "p1", "scripted-model", PROMPT_VERSION, fingerprint(make_paper().title, ABSTRACT)
    )
    assert restored is not None
    assert restored.origin is SummaryOrigin.EXTRACTIVE
    # ...but it must still report the method that actually wrote the text.
    assert restored.model == "extractive"


def test_an_unreadable_cache_row_is_a_miss_not_a_crash(cache: SummaryCache) -> None:
    cache._connection.execute(
        "INSERT INTO summaries (paper_id, model, prompt_version, source_fingerprint, "
        "text, origin, grounding, attempts, created_at) VALUES "
        "('p1', 'm1', 1, 'k', 'text', 'not-a-real-origin', '{bad json', 1, 'now')"
    )
    cache._connection.commit()
    assert cache.get("p1", "m1", 1, "k") is None


# --- extractive fallback --------------------------------------------------


def test_extractive_summaries_come_verbatim_from_the_abstract() -> None:
    text = ExtractiveSummarizer().summarize("A title", ABSTRACT)
    for sentence in text.split(". "):
        assert sentence.strip(". ") in ABSTRACT


def test_extractive_summaries_are_grounded_by_construction() -> None:
    text = ExtractiveSummarizer().summarize("Prime editing efficiency", ABSTRACT)
    assert GroundingChecker().check(text, ABSTRACT).passed


def test_structured_abstract_labels_are_stripped() -> None:
    text = ExtractiveSummarizer().summarize(
        "A study",
        "Background: Prime editing is precise. Methods: We benchmarked PE2. "
        "Results: Efficiency reached 42%. Conclusions: It works well.",
    )
    assert not text.startswith(("Background", "Methods", "Results", "Conclusions"))


def test_crossref_style_run_together_labels_are_stripped() -> None:
    text = ExtractiveSummarizer().summarize(
        "JIVE", "Abstract Motivation In single-cell analysis, batch effects matter."
    )
    assert text.startswith("In single-cell")


def test_a_sentence_starting_with_a_section_word_keeps_its_subject() -> None:
    # "Results show that..." must not become "show that...".
    text = ExtractiveSummarizer().summarize(
        "A study", "Results show that the treatment worked. It was measured carefully."
    )
    assert text.startswith("Results show that")


def test_a_paper_with_no_abstract_falls_back_to_its_title() -> None:
    assert ExtractiveSummarizer().summarize("Only a title", None) == "Only a title"


# --- the Mistral provider -------------------------------------------------


@respx.mock
async def test_the_provider_returns_the_message_content() -> None:
    respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": " Some summary. "}}]}
        )
    )
    provider = MistralProvider("key", "mistral-small-latest")
    assert await provider.complete("system", "user") == "Some summary."
    await provider.aclose()


@respx.mock
async def test_the_provider_sends_both_prompts_and_the_key() -> None:
    route = respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok."}}]})
    )
    provider = MistralProvider("secret-key", "mistral-small-latest")
    await provider.complete("SYS", "USR")
    await provider.aclose()

    request = route.calls[0].request
    assert request.headers["Authorization"] == "Bearer secret-key"
    import json

    body = json.loads(request.content)
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["messages"][1] == {"role": "user", "content": "USR"}
    assert body["model"] == "mistral-small-latest"


@respx.mock
async def test_transient_provider_errors_are_retried() -> None:
    route = respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json={"choices": [{"message": {"content": "ok."}}]}),
        ]
    )
    provider = MistralProvider("key", max_retries=1)
    assert await provider.complete("s", "u") == "ok."
    await provider.aclose()
    assert route.call_count == 2


@respx.mock
async def test_an_auth_failure_is_not_retried() -> None:
    route = respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=httpx.Response(401, text="unauthorized")
    )
    provider = MistralProvider("bad-key", max_retries=2)
    with pytest.raises(LLMUnavailableError, match="401"):
        await provider.complete("s", "u")
    await provider.aclose()
    assert route.call_count == 1


@respx.mock
async def test_rate_limiting_is_reported_distinctly() -> None:
    respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=httpx.Response(429)
    )
    provider = MistralProvider("key", max_retries=0)
    with pytest.raises(LLMRateLimitedError):
        await provider.complete("s", "u")
    await provider.aclose()


@respx.mock
async def test_an_unexpected_response_shape_is_an_error_not_a_crash() -> None:
    respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"unexpected": "shape"})
    )
    provider = MistralProvider("key", max_retries=0)
    with pytest.raises(LLMUnavailableError, match="unexpected response shape"):
        await provider.complete("s", "u")
    await provider.aclose()


@respx.mock
async def test_an_empty_completion_is_rejected() -> None:
    respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "   "}}]})
    )
    provider = MistralProvider("key", max_retries=0)
    with pytest.raises(LLMUnavailableError, match="empty completion"):
        await provider.complete("s", "u")
    await provider.aclose()


def test_no_api_key_yields_no_provider_rather_than_an_error() -> None:
    # A portfolio project that cannot be cloned and run without a paid key
    # is a worse project.
    assert build_provider("mistral", None, "mistral-small-latest") is None
    assert build_provider("none", "key", "m") is None
    assert build_provider("unknown-vendor", "key", "m") is None


def test_a_configured_provider_with_a_key_is_built() -> None:
    provider = build_provider("mistral", "key", "mistral-small-latest")
    assert isinstance(provider, MistralProvider)
    assert provider.model_id == "mistral-small-latest"


@respx.mock
async def test_a_zero_quota_account_is_distinguished_from_throttling() -> None:
    """A valid key on an unactivated account is not a pacing problem.

    Mistral returns 429 for both, but advertises a limit of 0 req/min when
    the account has no allocation. Retrying that is guaranteed waste, and
    calling it "rate limited" sends someone off to tune backoff for a
    problem that has nothing to do with request pacing.
    """
    route = respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=httpx.Response(
            429,
            headers={"x-ratelimit-limit-req-minute": "0"},
            json={"message": "Rate limit exceeded", "type": "rate_limited"},
        )
    )
    provider = MistralProvider("valid-key", max_retries=3)
    with pytest.raises(LLMQuotaError, match="no inference quota"):
        await provider.complete("s", "u")
    await provider.aclose()
    assert route.call_count == 1  # not retried


@respx.mock
async def test_genuine_throttling_is_still_retried() -> None:
    route = respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, headers={"x-ratelimit-limit-req-minute": "60"}),
            httpx.Response(200, json={"choices": [{"message": {"content": "ok."}}]}),
        ]
    )
    provider = MistralProvider("key", max_retries=1)
    assert await provider.complete("s", "u") == "ok."
    await provider.aclose()
    assert route.call_count == 2


@respx.mock
async def test_a_quota_failure_still_yields_an_extractive_summary() -> None:
    # The whole degradation chain: no quota -> no generation -> the user
    # still gets a summary and their search results.
    respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=httpx.Response(429, headers={"x-ratelimit-limit-req-minute": "0"})
    )
    provider = MistralProvider("key", max_retries=0)
    papers, report = await PaperSummarizer(provider).summarize([make_paper()])
    await provider.aclose()

    assert papers[0].summary is not None
    assert papers[0].summary.origin is SummaryOrigin.EXTRACTIVE
    assert report.applied is True
