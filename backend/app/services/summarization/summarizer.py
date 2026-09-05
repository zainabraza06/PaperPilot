"""Orchestrates generate → check → correct → fall back.

The policy in one place, so it can be read and argued with:

1. Look in the cache. A hit is returned as-is; nothing regenerates.
2. Ask the provider for a summary.
3. Check it against the abstract.
4. If it failed, ask again *with the specific failures quoted back*. This
   is a correction, not a re-roll: re-running the same prompt at a higher
   temperature would be sampling for luck.
5. If it failed again, fall back to extractive text taken verbatim from
   the abstract, and label it as such. Showing a summary known to be
   ungrounded — even with a warning badge — is not an option a research
   tool should offer.
6. Cache whatever was accepted, keyed so a prompt or model change cannot
   serve it later.

Never raises. A provider outage costs the user their summaries, not their
search results.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from app.core.logging import get_logger
from app.models.paper import Paper
from app.models.summary import (
    GroundingIssue,
    GroundingStatus,
    GroundingVerdict,
    Summary,
    SummaryOrigin,
    SummaryReport,
)
from app.services.summarization.extractive import ExtractiveSummarizer
from app.services.summarization.grounding import GroundingChecker
from app.services.summarization.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_retry_prompt,
    build_user_prompt,
)
from app.services.summarization.provider import LLMError, LLMProvider
from app.storage.summary_cache import SummaryCache, fingerprint

logger = get_logger(__name__)


class PaperSummarizer:
    """Produces a grounded summary for every paper in a result set."""

    def __init__(
        self,
        provider: LLMProvider | None,
        *,
        checker: GroundingChecker | None = None,
        extractive: ExtractiveSummarizer | None = None,
        cache: SummaryCache | None = None,
        max_concurrent: int = 5,
        max_attempts: int = 2,
    ) -> None:
        """
        Args:
            max_concurrent: bound on in-flight provider calls. Firing 50
                requests at once is the fastest way to get rate limited,
                which costs more time than the concurrency saved.
            max_attempts: total generation attempts before falling back.
        """
        self._provider = provider
        self._checker = checker or GroundingChecker()
        self._extractive = extractive or ExtractiveSummarizer()
        self._cache = cache
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_attempts = max_attempts

    @property
    def model_id(self) -> str:
        return self._provider.model_id if self._provider else self._extractive.model_id

    async def summarize(
        self, papers: Sequence[Paper]
    ) -> tuple[list[Paper], SummaryReport]:
        """Attach a summary to every paper. Order is preserved."""
        if not papers:
            return list(papers), SummaryReport(
                applied=False, reason="no papers to summarize"
            )

        started = time.perf_counter()
        summaries = await asyncio.gather(*(self._for_paper(paper) for paper in papers))

        enriched = [
            paper.model_copy(update={"summary": summary})
            for paper, summary in zip(papers, summaries, strict=True)
        ]
        report = SummaryReport(
            applied=True,
            model=self.model_id,
            summarized=len(summaries),
            from_cache=sum(1 for s in summaries if s.cached),
            regenerated=sum(1 for s in summaries if s.origin is SummaryOrigin.REGENERATED),
            fell_back=sum(1 for s in summaries if s.origin is SummaryOrigin.EXTRACTIVE),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            reason=None
            if self._provider
            else "no LLM provider configured; summaries are extractive",
        )
        return enriched, report

    # --- one paper ---------------------------------------------------------

    async def _for_paper(self, paper: Paper) -> Summary:
        source_key = fingerprint(paper.title, paper.abstract)

        cached = await self._lookup(paper.id, source_key)
        if cached is not None:
            return cached

        summary = await self._produce(paper)
        await self._store(paper.id, source_key, summary)
        return summary

    async def _produce(self, paper: Paper) -> Summary:
        """Generate and verify, falling back when that cannot be done."""
        if self._provider is None or not paper.abstract:
            # With no abstract there is nothing to ground against, so a
            # generated summary could never be verified even in principle.
            return self._extractive_summary(paper)

        previous_text = ""
        issues: list[GroundingIssue] = []
        # Kept across attempts so a rescued summary still records what was
        # wrong with the first try - that is the signal worth measuring.
        first_rejection: list[GroundingIssue] = []
        for attempt in range(1, self._max_attempts + 1):
            prompt = (
                build_user_prompt(paper.title, paper.abstract)
                if attempt == 1
                else build_retry_prompt(paper.title, paper.abstract, previous_text, issues)
            )
            try:
                async with self._semaphore:
                    text = await self._provider.complete(SYSTEM_PROMPT, prompt)
            except LLMError as exc:
                logger.warning("summarization failed for %s: %s", paper.id, exc)
                return self._extractive_summary(paper)

            verdict = self._checker.check(text, paper.abstract)
            if verdict.passed:
                return Summary(
                    text=text,
                    origin=SummaryOrigin.GENERATED
                    if attempt == 1
                    else SummaryOrigin.REGENERATED,
                    grounding=verdict,
                    model=self._provider.model_id,
                    prompt_version=PROMPT_VERSION,
                    attempts=attempt,
                    rejected_for=first_rejection,
                )
            if not first_rejection:
                first_rejection = list(verdict.issues)
            previous_text, issues = text, verdict.issues
            logger.info(
                "summary for %s failed grounding on attempt %d: %s",
                paper.id,
                attempt,
                ", ".join(issue.kind.value for issue in issues),
            )

        return self._extractive_summary(
            paper, attempts=self._max_attempts, rejected_for=first_rejection
        )

    def _extractive_summary(
        self,
        paper: Paper,
        attempts: int = 1,
        rejected_for: list[GroundingIssue] | None = None,
    ) -> Summary:
        """Sentences from the abstract, grounded because they *are* the abstract."""
        text = self._extractive.summarize(paper.title, paper.abstract)
        status = (
            GroundingStatus.GROUNDED if paper.abstract else GroundingStatus.UNVERIFIABLE
        )
        return Summary(
            text=text,
            origin=SummaryOrigin.EXTRACTIVE,
            grounding=GroundingVerdict(status=status, overlap=1.0 if paper.abstract else 0.0),
            model=self._extractive.model_id,
            prompt_version=PROMPT_VERSION,
            attempts=attempts,
            rejected_for=list(rejected_for or []),
        )

    # --- cache -------------------------------------------------------------

    async def _lookup(self, paper_id: str, source_key: str) -> Summary | None:
        if self._cache is None:
            return None
        return await asyncio.to_thread(
            self._cache.get, paper_id, self.model_id, PROMPT_VERSION, source_key
        )

    async def _store(self, paper_id: str, source_key: str, summary: Summary) -> None:
        if self._cache is None:
            return
        try:
            await asyncio.to_thread(
                self._cache.put,
                paper_id,
                self.model_id,
                PROMPT_VERSION,
                source_key,
                summary,
            )
        except Exception:
            logger.exception("could not cache summary for %s", paper_id)
