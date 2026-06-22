"""
utils/llm_retry.py — Generic LLM retry with exponential backoff.

Handles rate-limiting and transient errors from any LangChain-compatible
LLM provider (AWS Bedrock, OpenAI, Anthropic, Ollama, etc.).
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any

log = logging.getLogger(__name__)

_MAX_RETRIES = 6
_BASE_DELAY = 5.0
_MAX_DELAY = 120.0

# Substrings that indicate a retryable condition (rate limit / overload)
_RETRYABLE_SUBSTRINGS = [
    "ThrottlingException",
    "TooManyRequestsException",
    "RequestLimitExceeded",
    "ServiceUnavailableException",
    "rate limit",
    "Rate limit",
    "Rate exceeded",
    "429",
    "overloaded",
    "server_error",
    "temporarily unavailable",
    "connection",
]


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception represents a retryable LLM error."""
    # 1. botocore ClientError (Bedrock)
    try:
        from botocore.exceptions import ClientError
        if isinstance(exc, ClientError):
            code = exc.response.get("Error", {}).get("Code", "")
            return code in (
                "ThrottlingException",
                "TooManyRequestsException",
                "RequestLimitExceeded",
                "ServiceUnavailableException",
            )
    except ImportError:
        pass

    # 2. openai.RateLimitError
    try:
        import openai
        if isinstance(exc, openai.RateLimitError):
            return True
    except ImportError:
        pass

    # 3. anthropic.OverloadedError
    try:
        import anthropic
        if isinstance(exc, anthropic.OverloadedError):
            return True
    except ImportError:
        pass

    # 4. String scan fallback
    msg = str(exc)
    return any(s in msg for s in _RETRYABLE_SUBSTRINGS)


async def with_llm_retry(
    coro_fn: Callable[[], Awaitable[Any]],
    max_retries: int = _MAX_RETRIES,
) -> Any:
    """
    Execute an async LLM call with exponential backoff on retryable errors.

    Parameters
    ----------
    coro_fn:
        Zero-argument async callable that invokes the LLM.
    max_retries:
        Maximum number of retry attempts (default 6).

    Returns
    -------
    Any
        The return value of coro_fn() on success.

    Raises
    ------
    The last exception if all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn()
        except Exception as exc:
            if not _is_retryable(exc):
                raise
            last_exc = exc
            if attempt == max_retries:
                break
            delay = min(_BASE_DELAY * (2 ** attempt) + random.uniform(0, _BASE_DELAY), _MAX_DELAY)
            log.warning(
                "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs.",
                attempt + 1, max_retries, exc, delay,
            )
            await asyncio.sleep(delay)

    log.error("LLM call failed after %d attempts: %s", max_retries, last_exc)
    raise last_exc


# Backward-compatible alias for existing callers still importing the old name
async def with_bedrock_retry(
    coro_fn: Callable[[], Awaitable[Any]],
    max_retries: int = _MAX_RETRIES,
) -> Any:
    """Deprecated alias for with_llm_retry. Use with_llm_retry instead."""
    return await with_llm_retry(coro_fn, max_retries=max_retries)
