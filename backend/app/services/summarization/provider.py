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
from typing import Any, Protocol, runtime_checkable

import httpx

from app.core.errors import PaperPilotError
from app.core.logging import get_logger

logger = get_logger(__name__)

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class LLMError(PaperPilotError):
    """The provider could not produce a completion."""


class LLMRateLimitedError(LLMError):
    """The provider is throttling us."""


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
                last_error = (
                    LLMRateLimitedError("Mistral is rate limiting requests")
                    if response.status_code == 429
                    else LLMUnavailableError(f"HTTP {response.status_code} from Mistral")
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
        return content.strip()


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
