"""LLM providers behind one narrow interface.

The interface is one method — take a system prompt and a user prompt,
return text — because that is genuinely all summarization needs, and a
wider interface would be a wider thing to reimplement per provider.
Everything that makes the feature trustworthy (the prompt, the grounding
check, the retry policy, the cache) lives above this line and is shared by
every provider.

``MistralProvider`` talks to the REST API with ``httpx`` rather than the
vendor SDK. That keeps the dependency surface identical to the source
connectors, makes the calls mockable with ``respx`` like everything else
in this codebase, and means the adapter is ~40 lines that any reader can
verify against the API docs.
"""

from __future__ import annotations

import asyncio
import random
import re
from typing import Any, Protocol, runtime_checkable

import httpx

from app.core.errors import PaperPilotError
from app.core.logging import get_logger

logger = get_logger(__name__)

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

#: Advertised requests-per-minute allowance. A value of 0 alongside a 429
#: means "no quota on this account", not "slow down".
_LIMIT_HEADER = "x-ratelimit-limit-req-minute"


class LLMError(PaperPilotError):
    """The provider could not produce a completion."""


class LLMRateLimitedError(LLMError):
    """The provider is throttling us. Retrying later may succeed."""


class LLMQuotaError(LLMError):
    """The account has no inference quota, so retrying can never succeed.

    Distinct from throttling because the remedy is different and the retry
    policy must be too. Mistral returns HTTP 429 for both, but advertises
    a rate limit of ``0`` requests per minute when a key is valid and the
    account simply has no allocation - which is what an unactivated free
    tier looks like. Retrying that is guaranteed waste, and reporting it as
    "rate limited" sends someone off to add backoff for a problem that has
    nothing to do with request pacing.
    """


class LLMUnavailableError(LLMError):
    """The provider returned an error or could not be reached."""


@runtime_checkable
class LLMProvider(Protocol):
    """Turns a pair of prompts into text."""

    #: Identifies the model in summaries, reports and cache keys.
    model_id: str

    async def complete(self, system: str, user: str, *, max_tokens: int = 300) -> str:
        """Return the model's completion, or raise an ``LLMError``."""
        ...


class MistralProvider:
    """Chat completions from the Mistral API."""

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-small-latest",
        *,
        base_url: str = "https://api.mistral.ai/v1",
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        temperature: float = 0.2,
    ) -> None:
        """
        Args:
            temperature: low by default. Summarization under a grounding
                constraint wants the most probable phrasing, not variety;
                sampling further from the abstract's own wording is exactly
                what produces ungrounded output.
        """
        self.model_id = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_retries = max_retries
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def complete(self, system: str, user: str, *, max_tokens: int = 300) -> str:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": self._temperature,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 2):
            try:
                response = await self._client.post(
                    f"{self._base_url}/chat/completions", json=payload, headers=headers
                )
            except httpx.TimeoutException as exc:
                last_error = LLMUnavailableError(f"request timed out: {exc}")
            except httpx.HTTPError as exc:
                last_error = LLMUnavailableError(f"transport error: {exc}")
            else:
                if response.status_code < 400:
                    return self._extract_text(response)
                if response.status_code not in _RETRYABLE_STATUS:
                    # 401/400 will fail identically on every retry.
                    raise LLMUnavailableError(
                        f"HTTP {response.status_code} from Mistral: {response.text[:200]}"
                    )
                if response.status_code == 429:
                    if _has_no_quota(response):
                        # Retrying a zero allowance is guaranteed waste.
                        raise LLMQuotaError(
                            "Mistral accepted the API key but the account has no "
                            "inference quota (rate limit is 0 requests/minute). "
                            "Activate a plan at https://console.mistral.ai/ - "
                            "retrying will not help."
                        )
                    last_error = LLMRateLimitedError(
                        f"Mistral is rate limiting requests: {response.text[:160]}"
                    )
                else:
                    last_error = LLMUnavailableError(
                        f"HTTP {response.status_code} from Mistral: {response.text[:160]}"
                    )

            if attempt <= self._max_retries:
                delay = min(2.0 ** (attempt - 1), 4.0) + random.uniform(0, 0.25)
                logger.warning("Mistral attempt %d failed (%s); retrying", attempt, last_error)
                await asyncio.sleep(delay)

        assert last_error is not None
        raise last_error

    @staticmethod
    def _extract_text(response: httpx.Response) -> str:
        """Pull the message content out, failing loudly on an unexpected shape."""
        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailableError(f"unexpected response shape: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMUnavailableError("provider returned an empty completion")
        return strip_markdown(content)


#: Markdown emphasis the model adds despite being told not to. Underscores
#: are deliberately left alone: they occur inside real identifiers
#: (``TP53_mutant``, ``log_2``) far more often than as italics here.
_EMPHASIS = re.compile(r"\*{1,3}(?=\S)(.+?)(?<=\S)\*{1,3}", re.DOTALL)
_CODE_SPAN = re.compile(r"`{1,3}(?=\S)(.+?)(?<=\S)`{1,3}", re.DOTALL)
_LEADING_MARKER = re.compile(r"^\s*(?:[-*+]\s+|#{1,6}\s+|>\s+)", re.MULTILINE)


def strip_markdown(text: str) -> str:
    """Remove markdown formatting a model added despite the instruction.

    A prompt is a request, not a guarantee, so the rule is enforced here as
    well as asked for. Without this a summary reaches the UI as
    "replaces **CRISPR-Cas9** with **Cas12a**" and renders with literal
    asterisks - or, worse, the frontend starts rendering model output as
    markdown to compensate, which is an injection surface.
    """
    cleaned = _LEADING_MARKER.sub("", text)
    cleaned = _EMPHASIS.sub(r"\1", cleaned)
    cleaned = _CODE_SPAN.sub(r"\1", cleaned)
    return " ".join(cleaned.split())


def _has_no_quota(response: httpx.Response) -> bool:
    """True when the provider advertises a zero requests-per-minute allowance."""
    raw = response.headers.get(_LIMIT_HEADER)
    if raw is None:
        return False
    try:
        return int(raw) == 0
    except ValueError:
        return False


def build_provider(
    provider: str | None, api_key: str | None, model: str
) -> LLMProvider | None:
    """Construct the configured provider, or ``None`` if none is usable.

    Returning ``None`` rather than raising is the point: with no API key
    the app still runs and still shows summaries, extractive ones, clearly
    labelled as such. A portfolio project that cannot be cloned and run
    without a paid key is a worse project.
    """
    if not provider or provider == "none":
        return None
    if not api_key:
        logger.warning(
            "LLM provider %r configured but no API key set; "
            "summaries will be extractive",
            provider,
        )
        return None
    if provider == "mistral":
        return MistralProvider(api_key, model)
    logger.warning("unknown LLM provider %r; summaries will be extractive", provider)
    return None
